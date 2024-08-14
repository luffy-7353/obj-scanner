package detection;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.CvMemStorage;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.opencv.face.Face;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_core.cvReleaseMemStorage;

public class FaceDetection {

    CascadeClassifier faceCascade;
    private CvMemStorage storage;
    private OpenCVFrameConverter.ToIplImage iplImageConverter;
    private OpenCVFrameConverter.ToMat toMatConverter;
   // public static final Logger log = LoggerFactory.getLogger(FaceDetection.class);

    public FaceDetection() {
        iplImageConverter = new OpenCVFrameConverter.ToIplImage();
        toMatConverter = new OpenCVFrameConverter.ToMat();

        try {
            File haarCascadeFile = new File(this.getClass().getResource("/detection/haar_face_detection_cascader.xml").toURI());
          //  log.debug("Using Haar Cascade file located at : {}", haarCascadeFile.getAbsolutePath());
            faceCascade = new CascadeClassifier(haarCascadeFile.getCanonicalPath());
            storage = CvMemStorage.create();

        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public Map<Rect, Mat> detect(Frame frame) {
        Map<Rect, Mat> detectFaces = new HashMap<>();
        RectVector detectObjects = new RectVector();

        Mat matImage = toMatConverter.convertToMat(frame);
        faceCascade.detectMultiScale(matImage, detectObjects);

        long nrOfPpl = detectObjects.size();
        for (int i = 0; i < nrOfPpl; i++) {

            Rect rect = detectObjects.get(i);
            Mat croppedMat = matImage.apply(new Rect(rect.x(), rect.y(), rect.width(), rect.height()));
            detectFaces.put(rect, croppedMat);
        }
        return detectFaces;
    }

    @Override
    public void finalize() {
        cvReleaseMemStorage(storage);
    }
}
