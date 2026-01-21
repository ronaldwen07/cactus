import 'download.dart';

/// Simple test script to demonstrate downloadModel functionality.
///
/// Run with: dart flutter/lib/utils/download_test.dart
void main() async {
  print('=== downloadModel Test ===\n');

  // Use a small public file for testing (GitHub raw file)
  const testUrl = 'https://raw.githubusercontent.com/cactus-compute/cactus/main/LICENSE';

  // Test 1: Download a small test file with progress
  print('Test 1: Download with progress callback');
  print('-' * 40);

  try {
    final path = await downloadModel(
      testUrl,
      filename: 'test-license.txt',
      onProgress: (progress, status) {
        final percentage = (progress * 100).toStringAsFixed(1);
        print('  [$percentage%] $status');
      },
    );
    print('  ✓ Downloaded to: $path\n');
  } catch (e) {
    print('  ✗ Error: $e\n');
  }

  // Test 2: Cache validation (should skip download)
  print('Test 2: Cache validation (should skip re-download)');
  print('-' * 40);

  try {
    final path = await downloadModel(
      testUrl,
      filename: 'test-license.txt',
      onProgress: (progress, status) {
        print('  [${(progress * 100).toStringAsFixed(1)}%] $status');
      },
    );
    print('  ✓ Path: $path\n');
  } catch (e) {
    print('  ✗ Error: $e\n');
  }

  // Test 3: Error handling (invalid URL - non-existent domain)
  print('Test 3: Error handling (invalid URL)');
  print('-' * 40);

  try {
    // Use timestamp to ensure unique filename (no cache hit)
    final uniqueName = 'should-fail-${DateTime.now().millisecondsSinceEpoch}.bin';
    await downloadModel(
      'https://this-domain-definitely-does-not-exist-xyz123.invalid/model.bin',
      filename: uniqueName,
      onProgress: (progress, status) {
        print('  [${(progress * 100).toStringAsFixed(1)}%] $status');
      },
    );
    print('  ✗ Should have thrown an error\n');
  } catch (e) {
    print('  ✓ Correctly threw error: ${e.runtimeType}\n');
  }

  // Test 4: Filename extraction from URL
  print('Test 4: Auto filename extraction from URL');
  print('-' * 40);

  try {
    final path = await downloadModel(
      'https://raw.githubusercontent.com/cactus-compute/cactus/main/README.md',
      // No filename provided - should extract 'README.md' from URL
      onProgress: (progress, status) {
        print('  [${(progress * 100).toStringAsFixed(1)}%] $status');
      },
    );
    print('  ✓ Auto-extracted filename, saved to: $path\n');
  } catch (e) {
    print('  ✗ Error: $e\n');
  }

  print('=== Tests Complete ===');
}
