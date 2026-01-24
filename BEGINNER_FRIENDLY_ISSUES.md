# Beginner-Friendly Open Issues in Cactus

This document lists open issues in the upstream [cactus-compute/cactus](https://github.com/cactus-compute/cactus) repository that are marked as **good first issue** and suitable for beginners.

## Summary

Found **2 open issues** labeled as "good first issue" in the cactus-compute/cactus repository. Both are part of a two-part effort to unify the model initialization API across Flutter and React Native platforms.

---

## Issue #96: Unify Model Initialization API by Removing Automatic Download from Flutter

**Status:** Open  
**Labels:** `good first issue`, `help wanted`  
**Link:** https://github.com/cactus-compute/cactus/issues/96  
**Comments:** 4  
**Created:** July 19, 2025  
**Last Updated:** January 21, 2026

### Problem Overview
There is a discrepancy in model initialization between Flutter and React Native (RN): Flutter's `CactusContext.init` automatically downloads models if `modelUrl` is provided, while RN requires manual handling of local paths. To unify, we need to remove auto-download from Flutter's init, making both platforms require a local path for consistency.

This keeps `init` lean and avoids forcing dependencies in RN (since RN does not ship with file management modules).

**Note**: This is part 1 of a two-part unification effort (remove auto-download from Flutter). See part 2 for adding an optional downloadModel utility: #97

### Task
- In `flutter/lib/context.dart`: Remove the download logic from `_resolveParams` (including `HttpClient` usage and URL handling)
- Update `CactusInitParams` to remove/deprecate fields like `modelUrl`, `modelFilename`, `mmprojUrl`, etc. (add deprecation warnings if keeping for backwards compatibility)
- Add validation in init: Throw a clear error if `modelPath` doesn't point to an existing local file
- Update the Flutter docs and example app to show manual download (using Dart's built-in `http` package) before calling `init`
- Test: Ensure init fails gracefully on invalid paths; verify no regressions in existing init flow (e.g., local paths still work)

### Contributor Checklist
- [ ] Review the Flutter codebase (focus on `context.dart`) to understand current download logic
- [ ] Make changes incrementally and test locally (e.g., run Flutter example app before/after)
- [ ] Handle edge cases: What if `modelPath` is invalid? What about large files or interrupted downloads (now user responsibility)?
- [ ] Update any related types/docs to match RN's `initLlama` params (e.g., only require model: string as local path)
- [ ] Run linter and tests; ensure no breaking changes without deprecation notices
- [ ] In PR description, please explain how this aligns with RN and link to part 2 issue

### PR Checklist
- [ ] API matches RN's `initLlama` (takes local path only, no URLs)
- [ ] Flutter `init` throws helpful errors for missing files
- [ ] Examples updated to demonstrate manual download + init sequence
- [ ] Marked as "good first issue" for new contributors—focus on Dart/Flutter basics

---

## Issue #97: Implement Optional downloadModel Utility in RN and Flutter for Consistency

**Status:** Open  
**Labels:** `good first issue`, `help wanted`  
**Link:** https://github.com/cactus-compute/cactus/issues/97  
**Comments:** 2  
**Created:** July 19, 2025  
**Last Updated:** January 21, 2026

### Problem Overview
There is a discrepancy in model initialization between Flutter and React Native (RN): Flutter's `CactusContext.init` automatically downloads models if `modelUrl` is provided, while RN requires manual handling of local paths. To unify, we need to remove auto-download from Flutter's init, making both platforms require a local path for consistency.

This keeps `init` lean and avoids forcing dependencies in RN (since RN does not ship with file management modules).

**Note**: This is part 2 of a two-part unification effort (remove auto-download from Flutter). See part 1 for removing auto-download from Flutter: #96

### Task

**Flutter** (`flutter/lib/utils/download.dart`):
- Create a new file and implement `Future<String> downloadModel(String url, {String? filename, Function(double, String)? onProgress})`
- Use Dart's built-in `http` package: Check if file exists in app docs dir; download if not; report progress via callback (e.g., percentage and status like "Downloading...")
- Return the local path on success

**React Native** (`src/utils/download.ts`):
- Create a new file and implement the same function signature using `react-native-fs` (treat as optional peer dependency and throw clear error like `"Install react-native-fs to use downloadModel"` if not found)
- Check if file exists; download to `DocumentDirectoryPath`; support progress callback
- In `src/index.ts`, export it with a try-catch for missing `rn-fs` (e.g., `export { downloadModel } from './utils/download';`)

- For both: Handle basics like custom filenames, retries on failure (simple 3-try logic), and caching (skip if file exists and is valid/non-zero size)
- Update docs and examples in both platforms to show `downloadModel` usage before `init` (e.g., `const path = await downloadModel(url); await init({ model: path });`)

### Contributor Checklist
- [ ] Install required deps if needed (Flutter: none; RN: `yarn add react-native-fs` for testing)
- [ ] Test on device/simulator: Download a small test file; verify progress callback works; check error handling (e.g., bad URL, no network)
- [ ] Keep in mind: No forced deps in RN core: utility should fail gracefully if `rn-fs` missing. Support large files (progress essential)
- [ ] Align signatures exactly between platforms (same params, return local path string)
- [ ] Update `README.md` with examples; ensure no changes to core init functions here (that's part 1)
- [ ] Run linter/tests; in PR, bonus points for demo with screenshots/logs of download working :)

### PR Checklist
- [ ] Functions return local path; progress callback fires correctly (e.g., 0-1.0 scale)
- [ ] RN version doesn't break if rn-fs not installed (throws helpful error)
- [ ] Examples show end-to-end: download + init
- [ ] Good for beginners with file/network handling; cross-links to part 1

---

## Why These Issues Are Good for Beginners

Both issues are well-documented with:
- Clear problem statements and context
- Detailed task descriptions broken down into actionable steps
- Comprehensive contributor and PR checklists
- Focus on specific file changes with exact file paths provided
- Good separation of concerns (part 1 and part 2)
- Marked explicitly as "good first issue" by the maintainers
- Opportunity to work with mobile development (Flutter and React Native)
- Clear testing requirements and expected outcomes

---

## How to Get Started

1. **Fork the upstream repository**: https://github.com/cactus-compute/cactus
2. **Read the issue carefully**: Understand the problem and the proposed solution
3. **Set up your development environment**: Follow the setup instructions in the main README
4. **Create a branch**: Use a descriptive branch name (e.g., `fix-issue-96-remove-flutter-download`)
5. **Make your changes**: Follow the task description and checklists
6. **Test thoroughly**: Run the Flutter/RN example apps to verify your changes work
7. **Submit a PR**: Reference the issue number in your PR description and explain your changes

---

## Additional Resources

- **Main Repository**: https://github.com/cactus-compute/cactus
- **Documentation**: Available in the `/docs` folder of the repository
- **Contributing Guide**: See `CONTRIBUTING.md` in the repository
- **Community**: Join the discussion in the issue comments if you have questions

---

*Document generated on: January 24, 2026*
