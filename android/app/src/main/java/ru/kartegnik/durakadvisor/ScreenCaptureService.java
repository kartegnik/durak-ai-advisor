package ru.kartegnik.durakadvisor;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.SystemClock;
import android.view.WindowManager;

import java.io.IOException;
import java.nio.ByteBuffer;

public final class ScreenCaptureService extends Service {
    static final String ACTION_START = "ru.kartegnik.durakadvisor.START";
    static final String ACTION_STOP = "ru.kartegnik.durakadvisor.STOP";
    static final String ACTION_RESET = "ru.kartegnik.durakadvisor.RESET";
    static final String EXTRA_RESULT_CODE = "result_code";
    static final String EXTRA_RESULT_DATA = "result_data";

    private static final String CHANNEL_ID = "screen_analysis";
    private static final int NOTIFICATION_ID = 41;
    private static final long FRAME_INTERVAL_MS = 300L;
    private static volatile boolean running;

    private HandlerThread captureThread;
    private Handler captureHandler;
    private ImageReader imageReader;
    private VirtualDisplay virtualDisplay;
    private MediaProjection mediaProjection;
    private MediaProjection.Callback projectionCallback;
    private OverlayController overlay;
    private FrameAnalyzer analyzer;
    private int densityDpi;
    private long lastFrameAt;
    private boolean shuttingDown;

    static boolean isRunning() {
        return running;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
        overlay = new OverlayController(
                this, this::requestAnalyzerReset, this::requestManualSuit);
        try {
            analyzer = new FrameAnalyzer(this, overlay::show);
        } catch (IOException error) {
            stopSelf();
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            return START_NOT_STICKY;
        }
        if (ACTION_STOP.equals(intent.getAction())) {
            stopSelf();
            return START_NOT_STICKY;
        }
        if (ACTION_RESET.equals(intent.getAction())) {
            requestAnalyzerReset();
            return START_NOT_STICKY;
        }
        if (!ACTION_START.equals(intent.getAction()) || analyzer == null) {
            return START_NOT_STICKY;
        }

        startAsForeground();
        if (mediaProjection == null) {
            int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, ActivityResultCodes.CANCELED);
            Intent resultData = readResultData(intent);
            if (resultCode != ActivityResultCodes.OK || resultData == null) {
                stopSelf();
                return START_NOT_STICKY;
            }
            startProjection(resultCode, resultData);
        }
        return START_NOT_STICKY;
    }

    private void requestAnalyzerReset() {
        Handler handler = captureHandler;
        if (handler == null || analyzer == null) {
            return;
        }
        handler.post(() -> {
            analyzer.reset();
            lastFrameAt = 0L;
            overlay.show("Ищу козырную карту…");
        });
    }

    private void requestManualSuit(String suit) {
        Handler handler = captureHandler;
        if (handler == null || analyzer == null) {
            return;
        }
        handler.post(() -> analyzer.lockSuit(suit));
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        shuttingDown = true;
        running = false;
        if (overlay != null) {
            overlay.close();
        }
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
        if (mediaProjection != null) {
            if (projectionCallback != null) {
                mediaProjection.unregisterCallback(projectionCallback);
            }
            mediaProjection.stop();
            mediaProjection = null;
        }
        if (captureThread != null) {
            captureThread.quitSafely();
            captureThread = null;
        }
        super.onDestroy();
    }

    private void startProjection(int resultCode, Intent resultData) {
        captureThread = new HandlerThread("durak-screen-analysis");
        captureThread.start();
        captureHandler = new Handler(captureThread.getLooper());

        MediaProjectionManager manager = (MediaProjectionManager) getSystemService(
                Context.MEDIA_PROJECTION_SERVICE);
        mediaProjection = manager.getMediaProjection(resultCode, resultData);
        projectionCallback = new MediaProjection.Callback() {
            @Override
            public void onStop() {
                if (!shuttingDown) {
                    stopSelf();
                }
            }

            @Override
            public void onCapturedContentResize(int width, int height) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                    captureHandler.post(() -> resizeCapture(width, height));
                }
            }
        };
        mediaProjection.registerCallback(projectionCallback, captureHandler);

        Rect bounds = currentBounds();
        densityDpi = getResources().getDisplayMetrics().densityDpi;
        imageReader = createImageReader(bounds.width(), bounds.height());
        virtualDisplay = mediaProjection.createVirtualDisplay(
                "DurakAdvisorCapture",
                bounds.width(), bounds.height(), densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, captureHandler);
        running = true;
        overlay.show("Ищу козырную карту…");
    }

    private ImageReader createImageReader(int width, int height) {
        ImageReader reader = ImageReader.newInstance(
                Math.max(1, width), Math.max(1, height), PixelFormat.RGBA_8888, 2);
        reader.setOnImageAvailableListener(this::onImageAvailable, captureHandler);
        return reader;
    }

    private void resizeCapture(int width, int height) {
        if (virtualDisplay == null || width <= 0 || height <= 0) {
            return;
        }
        ImageReader previous = imageReader;
        imageReader = createImageReader(width, height);
        virtualDisplay.resize(width, height, densityDpi);
        virtualDisplay.setSurface(imageReader.getSurface());
        if (previous != null) {
            previous.close();
        }
    }

    private void onImageAvailable(ImageReader reader) {
        Image image = reader.acquireLatestImage();
        if (image == null) {
            return;
        }
        try {
            long now = SystemClock.elapsedRealtime();
            if (now - lastFrameAt < FRAME_INTERVAL_MS) {
                return;
            }
            lastFrameAt = now;
            Bitmap bitmap = imageToBitmap(image);
            try {
                analyzer.analyze(bitmap);
            } finally {
                bitmap.recycle();
            }
        } finally {
            image.close();
        }
    }

    private static Bitmap imageToBitmap(Image image) {
        Image.Plane plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int width = image.getWidth();
        int height = image.getHeight();
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        int paddedWidth = width + (rowStride - pixelStride * width) / pixelStride;
        Bitmap padded = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888);
        padded.copyPixelsFromBuffer(buffer);
        if (paddedWidth == width) {
            return padded;
        }
        Bitmap cropped = Bitmap.createBitmap(padded, 0, 0, width, height);
        padded.recycle();
        return cropped;
    }

    private Rect currentBounds() {
        WindowManager manager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            return manager.getMaximumWindowMetrics().getBounds();
        }
        android.util.DisplayMetrics metrics = new android.util.DisplayMetrics();
        manager.getDefaultDisplay().getRealMetrics(metrics);
        return new Rect(0, 0, metrics.widthPixels, metrics.heightPixels);
    }

    private void startAsForeground() {
        Notification notification = notification();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                    NOTIFICATION_ID, notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
    }

    private Notification notification() {
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent openPending = PendingIntent.getActivity(
                this, 0, open, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Intent stop = new Intent(this, ScreenCaptureService.class);
        stop.setAction(ACTION_STOP);
        PendingIntent stopPending = PendingIntent.getService(
                this, 1, stop, PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        return new Notification.Builder(this, CHANNEL_ID)
                .setSmallIcon(android.R.drawable.ic_menu_view)
                .setContentTitle("Durak AI Advisor")
                .setContentText("Экран анализируется только в памяти")
                .setContentIntent(openPending)
                .setOngoing(true)
                .addAction(new Notification.Action.Builder(
                        android.R.drawable.ic_menu_close_clear_cancel,
                        "Остановить", stopPending).build())
                .build();
    }

    private void createNotificationChannel() {
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID, "Анализ экрана", NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("Показывает, когда включено распознавание игры");
        NotificationManager manager = (NotificationManager) getSystemService(
                Context.NOTIFICATION_SERVICE);
        manager.createNotificationChannel(channel);
    }

    private static Intent readResultData(Intent intent) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            return intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent.class);
        }
        return intent.getParcelableExtra(EXTRA_RESULT_DATA);
    }

    private static final class ActivityResultCodes {
        static final int OK = -1;
        static final int CANCELED = 0;

        private ActivityResultCodes() {}
    }
}
