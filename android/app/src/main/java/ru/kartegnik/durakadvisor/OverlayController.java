package ru.kartegnik.durakadvisor;

import android.content.Context;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.function.Consumer;

final class OverlayController {
    private final Context context;
    private final WindowManager windowManager;
    private final Runnable resetAction;
    private final Consumer<String> suitAction;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private LinearLayout view;
    private TextView statusView;
    private LinearLayout controlsView;

    OverlayController(Context context, Runnable resetAction, Consumer<String> suitAction) {
        this.context = context;
        this.resetAction = resetAction;
        this.suitAction = suitAction;
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
            view = new LinearLayout(context);
            view.setOrientation(LinearLayout.VERTICAL);
            view.setGravity(Gravity.CENTER_HORIZONTAL);
            int horizontal = dp(14);
            int vertical = dp(9);
            view.setPadding(horizontal, vertical, horizontal, vertical);

            GradientDrawable background = new GradientDrawable();
            background.setColor(0xE6183153);
            background.setCornerRadius(dp(12));
            view.setBackground(background);

            statusView = new TextView(context);
            statusView.setTextColor(Color.WHITE);
            statusView.setTextSize(14f);
            statusView.setGravity(Gravity.CENTER);
            statusView.setMaxWidth(
                    context.getResources().getDisplayMetrics().widthPixels * 9 / 10);
            view.addView(statusView, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT));

            controlsView = new LinearLayout(context);
            controlsView.setOrientation(LinearLayout.HORIZONTAL);
            controlsView.setGravity(Gravity.CENTER);
            view.addView(controlsView, new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT));

            addSuitButton("♥", "H", 0xffff7a7a, "Черви");
            addSuitButton("♦", "D", 0xffff7a7a, "Бубны");
            addSuitButton("♣", "C", Color.WHITE, "Крести");
            addSuitButton("♠", "S", Color.WHITE, "Пики");

            TextView reset = actionView("↻", 0xffb9ffcf);
            reset.setContentDescription("Повторить автоматический поиск козыря");
            reset.setOnClickListener(ignored -> resetAction.run());
            controlsView.addView(reset);

            WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.WRAP_CONTENT,
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                    WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                            | WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
                            | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                    PixelFormat.TRANSLUCENT);
            params.gravity = Gravity.TOP | Gravity.CENTER_HORIZONTAL;
            params.y = dp(4);
            windowManager.addView(view, params);
        }
        statusView.setText(text);
    }

    private void addSuitButton(String symbol, String suit, int color, String description) {
        TextView button = actionView(symbol, color);
        button.setContentDescription("Выбрать козырь: " + description);
        button.setOnClickListener(ignored -> suitAction.accept(suit));
        controlsView.addView(button);
    }

    private TextView actionView(String text, int color) {
        TextView button = new TextView(context);
        button.setText(text);
        button.setTextColor(color);
        button.setTextSize(21f);
        button.setGravity(Gravity.CENTER);
        button.setPadding(dp(6), dp(3), dp(6), dp(3));
        button.setLayoutParams(new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        return button;
    }

    private int dp(int value) {
        return Math.round(value * context.getResources().getDisplayMetrics().density);
    }
}
