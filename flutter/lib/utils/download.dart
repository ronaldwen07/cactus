import 'dart:io';
import 'dart:async';

/// Downloads a model file from a URL to the app's documents directory.
///
/// Parameters:
/// - [url]: The URL to download the model from
/// - [filename]: Optional custom filename. If not provided, extracts from URL
/// - [onProgress]: Optional callback for download progress updates
///   - First parameter: progress (0.0 to 1.0)
///   - Second parameter: status message
///
/// Returns the local file path on success.
///
/// Example:
/// ```dart
/// final modelPath = await downloadModel(
///   'https://huggingface.co/model/resolve/main/model.bin',
///   filename: 'my-model.bin',
///   onProgress: (progress, status) {
///     print('$status: ${(progress * 100).toStringAsFixed(1)}%');
///   },
/// );
/// final cactus = Cactus.create(modelPath);
/// ```
Future<String> downloadModel(
  String url, {
  String? filename,
  void Function(double progress, String status)? onProgress,
}) async {
  const maxRetries = 3;

  // Get app documents directory
  final documentsDir = await _getDocumentsDirectory();
  final modelsDir = Directory('${documentsDir.path}/models');

  // Create models directory if it doesn't exist
  if (!await modelsDir.exists()) {
    await modelsDir.create(recursive: true);
  }

  // Determine filename from URL if not provided
  final effectiveFilename = filename ?? _extractFilename(url);
  final filePath = '${modelsDir.path}/$effectiveFilename';
  final file = File(filePath);

  // Check if file already exists with non-zero size (cache validation)
  if (await file.exists()) {
    final fileSize = await file.length();
    if (fileSize > 0) {
      onProgress?.call(1.0, 'Using cached model');
      return filePath;
    }
  }

  // Download with retry logic
  Exception? lastException;
  for (int attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      onProgress?.call(0.0, 'Starting download (attempt $attempt/$maxRetries)');
      await _downloadFile(url, file, onProgress);
      onProgress?.call(1.0, 'Download complete');
      return filePath;
    } catch (e) {
      lastException = e is Exception ? e : Exception(e.toString());
      if (attempt < maxRetries) {
        onProgress?.call(0.0, 'Retry ${attempt + 1}/$maxRetries after error');
        await Future.delayed(Duration(seconds: attempt * 2));
      }
    }
  }

  throw lastException ?? Exception('Download failed after $maxRetries attempts');
}

/// Downloads a file from URL with progress reporting.
Future<void> _downloadFile(
  String url,
  File file,
  void Function(double progress, String status)? onProgress,
) async {
  final client = HttpClient();

  try {
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();

    if (response.statusCode != 200) {
      throw HttpException(
        'Failed to download: HTTP ${response.statusCode}',
        uri: Uri.parse(url),
      );
    }

    final contentLength = response.contentLength;
    int bytesReceived = 0;

    final sink = file.openWrite();

    try {
      await for (final chunk in response) {
        sink.add(chunk);
        bytesReceived += chunk.length;

        if (contentLength > 0 && onProgress != null) {
          final progress = bytesReceived / contentLength;
          final mbReceived = (bytesReceived / (1024 * 1024)).toStringAsFixed(1);
          final mbTotal = (contentLength / (1024 * 1024)).toStringAsFixed(1);
          onProgress(progress, 'Downloading: $mbReceived MB / $mbTotal MB');
        }
      }
    } finally {
      await sink.close();
    }
  } finally {
    client.close();
  }
}

/// Extracts filename from a URL.
String _extractFilename(String url) {
  final uri = Uri.parse(url);
  final pathSegments = uri.pathSegments;

  if (pathSegments.isNotEmpty) {
    final lastSegment = pathSegments.last;
    if (lastSegment.isNotEmpty) {
      return lastSegment;
    }
  }

  // Fallback to a hash-based name if URL doesn't contain a clear filename
  return 'model_${url.hashCode.abs()}.bin';
}

/// Gets the application documents directory.
Future<Directory> _getDocumentsDirectory() async {
  // On mobile platforms, we use platform-specific paths
  // On desktop/other platforms, fall back to current directory
  if (Platform.isIOS || Platform.isAndroid) {
    // For mobile, use the app's documents directory
    // This requires path_provider package in a full Flutter app
    // For standalone Dart, we use a reasonable default
    final home = Platform.environment['HOME'] ?? '.';
    return Directory('$home/Documents');
  } else if (Platform.isMacOS) {
    final home = Platform.environment['HOME'] ?? '.';
    return Directory('$home/Library/Application Support/Cactus');
  } else if (Platform.isWindows) {
    final appData = Platform.environment['APPDATA'] ?? '.';
    return Directory('$appData/Cactus');
  } else if (Platform.isLinux) {
    final home = Platform.environment['HOME'] ?? '.';
    return Directory('$home/.local/share/cactus');
  }

  return Directory.current;
}
