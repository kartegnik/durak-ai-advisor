# Android prototype

The Android client processes `MediaProjection` frames only in memory. It does
not request Internet or storage permissions and does not write screenshots to
disk.

Current milestone (`0.3.0`):

- asks for Android's screen-capture consent;
- runs capture in a `mediaProjection` foreground service;
- shows an overlay above the game with manual suit buttons and automatic retry;
- recognizes and locks the trump suit after two matching frames;
- recognizes cards in the player's hand and on the table with the exported
  YOLO localizer and 36-class ensemble classifier;
- remembers played, taken and discarded cards across frames and infers which
  side is attacking from the first card of each bout;
- generates only rule-legal moves for attack, defense and transfer Durak;
- scores those moves on-device with the exported recurrent PPO v3 policy;
- tries the complete screen plus top-aligned and centered reference-aspect
  viewports, so navigation bars and taller displays do not require a fixed
  resolution.

The overlay reports the directly observed hand and table plus one recommended
move. Before the first table card it prefixes the recommendation with
"If it is your turn", because the screen alone does not reveal the opening
player until somebody moves.

## Build

Open the `android` directory in Android Studio, let it install Android SDK 35,
and build the `app` configuration. A command-line build will also be available
through `./gradlew assembleDebug` once the Gradle wrapper is generated.

## Run

1. Install and open the app.
2. Grant "display over other apps" permission.
3. Tap "Start screen analysis" and accept Android's screen-capture dialog.
4. Open a fresh game, tap the correct trump suit in the overlay, and leave the
   initial hand visible for two analysis frames.
5. Tap the circular-arrow button whenever a new game begins. This resets the
   remembered cards and the recurrent policy state.
6. Stop capture from the app or its persistent notification.
