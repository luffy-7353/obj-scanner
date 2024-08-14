import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.opencv.core.CvType;

import javax.swing.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;

import static org.bytedeco.javacv.Java2DFrameUtils.toBufferedImage;
import static org.bytedeco.opencv.global.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Main {
    public static void main(String[] args) {

        //OpenCV.loadShared();
        //ystem.loadLibrary(Core.NATIVE_LIBRARY_NAME);
       // System.out.println(Core.getVersionMajor());



        final Mat src = imread("D:\\akka-persistence-actor\\learn-opencv\\src\\main\\resources\\data\\20230814_094715.jpg");
        display(src, "Input");

        // Apply Laplacian filter
        final Mat dest = new Mat();
        final Mat dest2 = new Mat();
        Laplacian(src, dest, src.depth(), 1, 3, 0, BORDER_DEFAULT);
        display(dest, "Laplacian");

        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        //VideoCapture capture = new VideoCapture();

    }

    static void display(Mat image, String caption) {
        // Create image window named "My Image".
        final CanvasFrame canvas = new CanvasFrame(caption, 1.0);

        // Request closing of the application when the image window is closed.
        canvas.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        // Convert from OpenCV Mat to Java Buffered image for display
        final BufferedImage bi = toBufferedImage(image);

        // Show image on window.
        canvas.showImage(bi);
    }
}