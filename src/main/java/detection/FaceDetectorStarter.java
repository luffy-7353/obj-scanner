package detection;


import net.coobird.thumbnailator.Thumbnails;
import org.bytedeco.javacv.*;
import org.bytedeco.javacv.Frame;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;

import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgproc.*;


public class FaceDetectorStarter {

    //public static final Logger log = LoggerFactory.getLogger(FaceDetectorStarter.class);
    private FrameGrabber frameGrabber;
    private FaceDetection faceDetection = new FaceDetection();
    private static Java2DFrameConverter frameConverter = new Java2DFrameConverter();

    private OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private static JFrame window;
    private JPanel videoPanel;
    private volatile boolean running = false;

    public FaceDetectorStarter() {
        window = new JFrame();
        videoPanel = new JPanel();

        window.setLayout(new BorderLayout());
        window.setSize(new Dimension(1280, 720));
        window.add(videoPanel, BorderLayout.CENTER);
        window.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                stop();
            }
        });
    }

    public void process() {
        running = true;
        try {
            frameGrabber.trigger();
            org.bytedeco.javacv.Frame frame = frameGrabber.grab();


            while ((frame = frameGrabber.grab()) != null) {
                Map<Rect, Mat> detectedFaces = faceDetection.detect(frame);
                Mat mat = toMatConverter.convert(frame);

                detectedFaces.entrySet().forEach(rectMatEntry -> {

                /*String age = ageDetector.predictAge(rectMatEntry.getValue(), frame);
                CNNGenderDetector.Gender gender = genderDetector.predictGender(rectMatEntry.getValue(), frame);
                String caption = String.format("%s:[%s]", gender, age);
                logger.debug("Face's caption : {}", caption);


*/
                    rectangle(mat,
                            new Point(rectMatEntry.getKey().x(), rectMatEntry.getKey().y()),
                            new Point(rectMatEntry.getKey().width() + rectMatEntry.getKey().x(), rectMatEntry.getKey().height() + rectMatEntry.getKey().y()),
                            Scalar.RED,
                            2,
                            CV_AA,
                            0);

                    int posX = Math.max(rectMatEntry.getKey().x() - 10, 0);
                    int posY = Math.max(rectMatEntry.getKey().y() - 10, 0);
                    putText(mat, "tara", new Point(posX, posY), CV_FONT_HERSHEY_PLAIN, 1.0,
                            new Scalar(255, 255, 255, 2.0));
                });

                // Show the processed mat in UI
                Frame processedFrame = toMatConverter.convert(mat);

                Graphics graphics = videoPanel.getGraphics();
                BufferedImage resizedImage = getResizedBufferedImage(processedFrame, videoPanel);
                SwingUtilities.invokeLater(() -> {
                    graphics.drawImage(resizedImage, 0, 0, videoPanel);
                });
            }


        } catch (FrameGrabber.Exception e) {
            // log.error("Exception while grabbing frame {}", e.getMessage());
        }
    }

    public void start() {
        frameGrabber = new OpenCVFrameGrabber(0);
        // frameGrabber = new VideoInputFrameGrabber(0);
        frameGrabber.setImageWidth(1280);
        frameGrabber.setImageHeight(720);

        // log.info("Starting frame grabber");
        try {
            frameGrabber.start();
            //   log.info("Started frame grabber with image width-height : {}-{}", frameGrabber.getImageWidth(), frameGrabber.getImageHeight());
        } catch (Exception ex) {
            // log.error("Exception while starting the face detection starter {}", ex.getMessage());
        }

        SwingUtilities.invokeLater(() -> {
            window.setVisible(true);
        });

        process();
    }

    public void stop() {
        running = false;
        try {
            // log.info("Releasing and stopping FrameGrabber");
            frameGrabber.release();
            frameGrabber.stop();
        } catch (FrameGrabber.Exception e) {
            //log.error("Error occurred when stopping the FrameGrabber", e);
        }

        window.dispose();
    }

    public static void main(String[] args) {
        FaceDetectorStarter starter = new FaceDetectorStarter();
        //log.info("Starter invoked");
        new Thread(starter::start).start();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            //   log.info("Stopping javacv example");
            starter.stop();
        }));

        try {
            Thread.currentThread().join();
        } catch (InterruptedException ignored) {
        }
    }

    public static BufferedImage getResizedBufferedImage(Frame frame, JPanel videoPanel) {
        BufferedImage resizedImage = null;

        try {
            /*
             * We get notified about the frames that are being added. Then we pass each frame to BufferedImage. I have used
             * a library called Thumbnailator to achieve the resizing effect along with performance
             */
            resizedImage = Thumbnails.of(frameConverter.getBufferedImage(frame))
                    .size(videoPanel.getWidth(), videoPanel.getHeight())
                    .asBufferedImage();
        } catch (IOException e) {
            // log.error("Unable to convert the image to a buffered image", e);
        }

        return resizedImage;
    }
}
