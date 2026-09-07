# Android prototype

The Android client processes `MediaProjection` frames only in memory. It does
not request Internet or storage permissions and does not write screenshots to
disk.

Current milestone (`0.1.0`):

- asks for Android's screen-capture consent;
- runs capture in a `mediaProjection` foreground service;
- shows an overlay above the game with manual suit buttons and automatic retry;
- recognizes and locks the trump suit after two matching frames;
- tries the complete screen plus top-aligned and centered reference-aspect
  viewports, so navigation bars and taller displays do not require a fixed
  resolution.

Card recognition, round tracking, and neural move advice are the next
milestones. The overlay currently reports only the trump suit.

## Build

Open the `android` directory in Android Studio, let it install Android SDK 35,
and build the `app` configuration. A command-line build will also be available
through `./gradlew assembleDebug` once the Gradle wrapper is generated.

## Run

1. Install and open the app.
2. Grant "display over other apps" permission.
3. Tap "Start screen analysis" and accept Android's screen-capture dialog.
4. Open a fresh game while its face-up trump card is visible.
5. Stop capture from the app or its persistent notification.
