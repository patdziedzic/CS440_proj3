/**
 * 1-hot vector representing the color of a pixel
 */
public class PixelVector {
    double[] vector;

    PixelVector() {
        vector = new double[]{0, 0, 0, 0};
    }

    PixelVector(double a) {
        vector = new double[]{a};
    }

    PixelVector(double a, double b, double c, double d) {
        vector = new double[]{a, b, c, d};
    }

    PixelVector(PixelVector src) {
        vector = new double[src.vector.length];
        System.arraycopy(src.vector, 0, vector, 0, src.vector.length);
    }
}
