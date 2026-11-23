<?php
declare(strict_types=1);

/**
 * Streaming HTTPS reverse proxy for draft.bertholomus.com -> FrankenNAS origin via Cloudflare Tunnel.
 * Secrets are stored outside public_html (/home/u320837031/draft-proxy-secret.txt, mode 600).
 * No uploads are stored on shared hosting; all bytes stream directly to FrankenNAS /tank storage.
 */

@ini_set('display_errors', '0');
@ini_set('log_errors', '1');
@set_time_limit(300);

const ORIGIN_BASE = 'https://draft-origin.albertlee.co';
const SECRET_PATH = '/home/u320837031/draft-proxy-secret.txt';
const MAX_PROXY_BODY_BYTES = 16777216; // 16 MiB per single HTTP request (chunked uploads use 8 MiB chunks)

function fail_json(int $code, string $msg): void {
    http_response_code($code);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    echo json_encode(['error' => $msg, 'proxy' => 'draft.bertholomus.com']);
    exit;
}

if (!is_readable(SECRET_PATH)) {
    fail_json(503, 'origin auth configuration unavailable');
}
$originSecret = trim((string)file_get_contents(SECRET_PATH));
if ($originSecret === '') {
    fail_json(503, 'origin auth configuration empty');
}

$method = $_SERVER['REQUEST_METHOD'] ?? 'GET';
$allowedMethods = ['GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'OPTIONS'];
if (!in_array($method, $allowedMethods, true)) {
    fail_json(405, 'method not allowed');
}

$requestUri = $_SERVER['REQUEST_URI'] ?? '/';
if ($requestUri === '' || $requestUri[0] !== '/') {
    $requestUri = '/' . ltrim($requestUri, '/');
}
$targetUrl = ORIGIN_BASE . $requestUri;

$contentLength = isset($_SERVER['CONTENT_LENGTH']) ? (int)$_SERVER['CONTENT_LENGTH'] : 0;
if ($contentLength > MAX_PROXY_BODY_BYTES) {
    fail_json(413, 'single request body exceeds 16 MiB proxy limit; use chunked upload API');
}

// Gather headers to forward
$forwardHeaders = [
    'X-Origin-Auth: ' . $originSecret,
    'X-Forwarded-Host: draft.bertholomus.com',
    'X-Forwarded-Proto: https',
    'X-Forwarded-For: ' . ($_SERVER['REMOTE_ADDR'] ?? ''),
    'Expect:', // prevent 100-continue stalls
];

$passHeaderMap = [
    'HTTP_X_QUANT' => 'X-Quant',
    'HTTP_X_CONTENT_SHA256' => 'X-Content-SHA256',
    'HTTP_X_CHUNK_SHA256' => 'X-Chunk-SHA256',
    'HTTP_X_ARTIFACT_TYPE' => 'X-Artifact-Type',
    'HTTP_X_MANIFEST_JSON' => 'X-Manifest-JSON',
    'HTTP_X_REQUEST_ID' => 'X-Request-ID',
    'HTTP_X_CSRF_TOKEN' => 'X-CSRF-Token',
    'HTTP_COOKIE' => 'Cookie',
    'HTTP_ORIGIN' => 'Origin',
    'HTTP_REFERER' => 'Referer',
    'HTTP_USER_AGENT' => 'User-Agent',
    'HTTP_ACCEPT' => 'Accept',
    'HTTP_RANGE' => 'Range',
    'HTTP_IF_NONE_MATCH' => 'If-None-Match',
    'CONTENT_TYPE' => 'Content-Type',
];
$forwardedLower = [];
foreach ($passHeaderMap as $serverKey => $headerName) {
    if (isset($_SERVER[$serverKey]) && $_SERVER[$serverKey] !== '') {
        $val = str_replace(["\r", "\n"], '', (string)$_SERVER[$serverKey]);
        $forwardHeaders[] = $headerName . ': ' . $val;
        $forwardedLower[strtolower($headerName)] = true;
    }
}

$auth = $_SERVER['HTTP_AUTHORIZATION'] ?? $_SERVER['REDIRECT_HTTP_AUTHORIZATION'] ?? '';
if ($auth !== '') {
    $auth = str_replace(["\r", "\n"], '', (string)$auth);
    $forwardHeaders[] = 'Authorization: ' . $auth;
    $forwardedLower['authorization'] = true;
}

if (function_exists('getallheaders')) {
    $allHeaders = getallheaders();
    if (is_array($allHeaders)) {
        $fallbackTargets = [
            'authorization' => 'Authorization',
            'cookie' => 'Cookie',
            'x-csrf-token' => 'X-CSRF-Token',
        ];
        foreach ($allHeaders as $k => $v) {
            $lk = strtolower((string)$k);
            if (isset($fallbackTargets[$lk]) && empty($forwardedLower[$lk])) {
                $val = str_replace(["\r", "\n"], '', (string)$v);
                if ($val !== '') {
                    $forwardHeaders[] = $fallbackTargets[$lk] . ': ' . $val;
                    $forwardedLower[$lk] = true;
                }
            }
        }
    }
}

$ch = curl_init($targetUrl);
if ($ch === false) {
    fail_json(502, 'upstream transport init failed');
}

$bodyStream = null;
$bodyLen = 0;
if ($method === 'POST' || $method === 'PUT') {
    $in = fopen('php://input', 'rb');
    $tmp = fopen('php://temp/maxmemory:2097152', 'w+b');
    if ($in === false || $tmp === false) {
        fail_json(500, 'failed to allocate request stream');
    }
    while (!feof($in)) {
        $chunk = fread($in, 65536);
        if ($chunk === false) {
            break;
        }
        $bodyLen += strlen($chunk);
        if ($bodyLen > MAX_PROXY_BODY_BYTES) {
            fclose($in);
            fclose($tmp);
            curl_close($ch);
            fail_json(413, 'request body exceeds 16 MiB proxy limit; use chunked upload API');
        }
        fwrite($tmp, $chunk);
    }
    fclose($in);
    rewind($tmp);
    $bodyStream = $tmp;
}

$headersSent = false;
$statusCode = 200;
$hopByHop = [
    'connection' => true,
    'keep-alive' => true,
    'proxy-authenticate' => true,
    'proxy-authorization' => true,
    'te' => true,
    'trailers' => true,
    'transfer-encoding' => true,
    'upgrade' => true,
    'server' => true,
    'cf-ray' => true,
    'cf-cache-status' => true,
    'alt-svc' => true,
    'report-to' => true,
    'nel' => true,
    'content-encoding' => true,
];

curl_setopt($ch, CURLOPT_CUSTOMREQUEST, $method);
curl_setopt($ch, CURLOPT_HTTPHEADER, $forwardHeaders);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, false);
curl_setopt($ch, CURLOPT_FOLLOWLOCATION, false);
curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 15);
curl_setopt($ch, CURLOPT_TIMEOUT, 240);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, true);
curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, 2);

if ($method === 'HEAD') {
    curl_setopt($ch, CURLOPT_NOBODY, true);
}
if ($bodyStream !== null && $bodyLen > 0) {
    curl_setopt($ch, CURLOPT_UPLOAD, true);
    curl_setopt($ch, CURLOPT_INFILE, $bodyStream);
    curl_setopt($ch, CURLOPT_INFILESIZE, $bodyLen);
} elseif ($method === 'POST' || $method === 'PUT') {
    $forwardHeaders[] = 'Content-Length: 0';
    curl_setopt($ch, CURLOPT_HTTPHEADER, $forwardHeaders);
}

curl_setopt($ch, CURLOPT_HEADERFUNCTION, static function ($ch, string $line) use (&$statusCode, &$headersSent, $hopByHop): int {
    $len = strlen($line);
    $trimmed = trim($line);
    if ($trimmed === '') {
        return $len;
    }
    if (stripos($trimmed, 'HTTP/') === 0) {
        $parts = explode(' ', $trimmed, 3);
        if (isset($parts[1]) && is_numeric($parts[1])) {
            $statusCode = (int)$parts[1];
            http_response_code($statusCode);
        }
        return $len;
    }
    $colon = strpos($trimmed, ':');
    if ($colon !== false) {
        $name = trim(substr($trimmed, 0, $colon));
        $lower = strtolower($name);
        if (!isset($hopByHop[$lower])) {
            header($name . ': ' . trim(substr($trimmed, $colon + 1)), false, $statusCode);
            $headersSent = true;
        }
    }
    return $len;
});

curl_setopt($ch, CURLOPT_WRITEFUNCTION, static function ($ch, string $data): int {
    echo $data;
    if (ob_get_level() > 0) {
        @ob_flush();
    }
    flush();
    return strlen($data);
});

$ok = curl_exec($ch);
$curlErrNo = curl_errno($ch);
curl_close($ch);
if ($bodyStream !== null) {
    fclose($bodyStream);
}

if (!$ok && !$headersSent) {
    fail_json(502, 'origin gateway unreachable (code ' . $curlErrNo . ')');
}
