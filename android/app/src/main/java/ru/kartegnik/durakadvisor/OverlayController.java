package ru.kartegnik.durakadvisor;

import android.content.Context;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.TextView;

final class OverlayController {
    private final Context context;
    private final WindowManager windowManager;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private TextView view;

    OverlayController(Context context) {
        this.context = context;
        this.windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
    }

    void show(String text) {
        mainHandler.post(() -> showOnMainThread(text));
    }

    void close() {
        mainHandler.post(() -> {
            if (view != null) {
                windowManager.removeView(view);
                view = null;
            }
        });
    }

    private void showOnMainThread(String text) {
        if (!Settings.canDrawOverlays(context)) {
            return;
        }
        if (view == null) {
            view = new TextView(context);
            view.setTextColor(Color.WHITE);
            view.setTextSize(16f);
            view.setGravity(Gravity.CENTER);
            int horizontal = dp(14);
            int vertical = dp(9);
            view.setPadding(horizontal, vertical, horizontal, vertical);

            GradientDrawable background = new GradientDrawable();
            background.setColor(0xE6183153);
            background.setCornerRadius(dp(12));
            view.setBackground(background);

            WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                            | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                            | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                    PixelFormat.TRANSLUCENT);
            params.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
            params.y = dp(20);
            windowManager.addView(view, params);
        }
        view.setText(text);
    }

    private int dp(int value) {
        return Math.round(value * context.getResources().getDisplayMetrics().density);
    }
}
