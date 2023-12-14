import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;

public class Task2 extends Main {
    public static double trainingLossCutoff; //to prevent overfitting
    public static double divergenceCutoff; //to display that the data diverges
    public static double numTimesTrainAtOnce; //the number of times to train the model before calculating loss
    public static int numTrain;
    private static LinkedList<Image> trainingImages = new LinkedList<>();
    public static int numTest;
    private static LinkedList<Image> testingImages = new LinkedList<>();
    private static final double alpha = 0.015;
    private static final double initWeight = 0.001;
    private static PixelVector[] weightVector_R = new PixelVector[Main.D*Main.D + 1];
    private static PixelVector[] weightVector_G = new PixelVector[Main.D*Main.D + 1];
    private static PixelVector[] weightVector_B = new PixelVector[Main.D*Main.D + 1];
    private static PixelVector[] weightVector_Y = new PixelVector[Main.D*Main.D + 1];
    private static double softmaxDenominator;

    private static final HashMap<String, PixelVector> colors = new HashMap<String, PixelVector>(){{
        put("R", new PixelVector(1, 0, 0, 0));
        put("G", new PixelVector(0, 1, 0, 0));
        put("B", new PixelVector(0, 0, 1, 0));
        put("Y", new PixelVector(0, 0, 0, 1));
    }};


    /**
     * Get random int between min to max
     * @return randomly generated int
     */
    public static int rand(int min, int max) {
        return (int) ((Math.random() * (max - min)) + min);
    }

    /**
     * Softmax function, given that the denominator has been calculated already
     */
    public static double softmax(String color, PixelVector[] image) {
        //calculateSoftmaxDenominator();
        return (Math.exp(dotProduct(color, image))) / softmaxDenominator;
    }

    /**
     * Calculate the softmax denominator to normalize the function
     */
    public static void calculateSoftmaxDenominator(PixelVector[] image) {
        softmaxDenominator = 0.0;
        for (String color : colors.keySet()) {
            softmaxDenominator += Math.exp(dotProduct(color, image));
        }
    }

    /**
     * Dot product of weight (of given color) with given image vector
     * @param color The current color in iteration
     * @param image The PixelVector[] image used in calculations, also known as the x vector
     */
    public static double dotProduct(String color, PixelVector[] image) {
        PixelVector[] w = switch (color) {
            case "R" -> weightVector_R;
            case "G" -> weightVector_G;
            case "B" -> weightVector_B;
            case "Y" -> weightVector_Y;
            default -> null;
        };
        assert w != null;

        double sum = 0;
        //go through all vectors
        for (int i = 0; i < image.length; i++) {
            //go through all bits in the vector
            for (int j = 0; j < image[i].vector.length; j++)
                sum += w[i].vector[j] * image[i].vector[j];
        }
        return sum;
    }

    /**
     * Scalar multiplication for an Image
     */
    private static PixelVector[] scaleImage(double scalar, PixelVector[] img) {
        PixelVector[] resultImg = new PixelVector[img.length];
        for (int i = 0; i < resultImg.length; i++) {
            resultImg[i] = new PixelVector();
        }
        //go through all vectors
        for (int i = 0; i < img.length; i++) {
            resultImg[i].vector = new double[img[i].vector.length];
            //go through all bits in the vector
            for (int j = 0; j < img[i].vector.length; j++)
                resultImg[i].vector[j] = scalar * img[i].vector[j];
        }
        return resultImg;
    }

    /**
     * Subtract 2 vectors
     */
    private static PixelVector[] subtractVectors(PixelVector[] a, PixelVector[] b) {
        assert a.length == b.length;
        PixelVector[] result = new PixelVector[a.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = new PixelVector();
        }

        //go through all vectors
        for (int i = 0; i < result.length; i++) {
            result[i].vector = new double[a[i].vector.length];
            //go through all bits in the vector
            for (int j = 0; j < result[i].vector.length; j++)
                result[i].vector[j] = a[i].vector[j] - b[i].vector[j];
        }
        return result;
    }



    /**
     * TRAINING loss
     */
    private static double calculateTrainingLoss() {
        double sum = 0.0;
        for (Image img : trainingImages) {
            String y = img.colorToCut; //the actual output
            calculateSoftmaxDenominator(img.image);
            sum += -1 * Math.log(softmax(y, img.image));
        }

        return ((double) 1 / trainingImages.size()) * sum;
    }

    /**
     * TESTING loss
     */
    private static double calculateTestingLoss() {
        double sum = 0.0;
        for (Image img : testingImages) {
            String y = img.colorToCut;
            calculateSoftmaxDenominator(img.image);
            sum += -1 * Math.log(softmax(y, img.image));
        }

        return ((double) 1 / testingImages.size()) * sum;
    }



    /**
     * Generate an image
     */
    private static Image generateImage() {
        PixelVector[] pvArray = new PixelVector[Main.D*Main.D + 1];
        for (int i = 0; i < pvArray.length; i++) {
            pvArray[i] = new PixelVector();
        }
        pvArray[0] = new PixelVector(1);

        ArrayList<String> chooseColor = new ArrayList<>();
        chooseColor.add("R");
        chooseColor.add("G");
        chooseColor.add("B");
        chooseColor.add("Y");
        Collections.shuffle(chooseColor); //initial shuffling
        //shuffle until R is before Y, making the image dangerous
        while (chooseColor.indexOf("R") >= chooseColor.indexOf("Y"))
            Collections.shuffle(chooseColor);

        //randomly choose 2 rows
        int randRow1 = rand(0, 19);
        int randRow2 = rand(0, 19);
        while (randRow2 == randRow1)
            randRow2 = rand(0, 19);

        //randomly choose 2 cols
        int randCol1 = rand(0, 19);
        int randCol2 = rand(0, 19);
        while (randCol2 == randCol1)
            randCol2 = rand(0, 19);

        //deep copy (new addresses) the hashmap into 4 different PixelVectors for each color
        PixelVector color0 = new PixelVector();
        System.arraycopy(colors.get(chooseColor.get(0)).vector, 0,
                color0.vector, 0, colors.get(chooseColor.get(0)).vector.length);
        PixelVector color1 = new PixelVector();
        System.arraycopy(colors.get(chooseColor.get(1)).vector, 0,
                color1.vector, 0, colors.get(chooseColor.get(1)).vector.length);
        PixelVector color2 = new PixelVector();
        System.arraycopy(colors.get(chooseColor.get(2)).vector, 0,
                color2.vector, 0, colors.get(chooseColor.get(2)).vector.length);
        PixelVector color3 = new PixelVector();
        System.arraycopy(colors.get(chooseColor.get(3)).vector, 0,
                color3.vector, 0, colors.get(chooseColor.get(3)).vector.length);

        //randomly pick to either start with row or col
        if (Math.random() < 0.5) {
            //do row first
            for (int c = 0; c < Main.D; c++)
                pvArray[randRow1*Main.D + 1 + c] = new PixelVector(color0);
            for (int r = 0; r < Main.D*Main.D; r += Main.D)
                pvArray[randCol1 + 1 + r] = new PixelVector(color1);

            for (int c = 0; c < Main.D; c++)
                pvArray[randRow2 + 1 + c] = new PixelVector(color2);
            for (int r = 0; r < Main.D*Main.D; r += Main.D)
                pvArray[randCol2 + 1 + r] = new PixelVector(color3);
        }
        else {
            //do col first
            for (int r = 0; r < Main.D*Main.D; r += Main.D)
                pvArray[randCol1 + 1 + r] = new PixelVector(color0);
            for (int c = 0; c < Main.D; c++)
                pvArray[randRow1*Main.D + 1 + c] = new PixelVector(color1);

            for (int r = 0; r < Main.D*Main.D; r += Main.D)
                pvArray[randCol2 + 1 + r] = new PixelVector(color2);
            for (int c = 0; c < Main.D; c++)
                pvArray[randRow2 + 1 + c] = new PixelVector(color3);
        }
        Image img = new Image();
        img.image = pvArray;
        img.dangerous = chooseColor.indexOf("R") < chooseColor.indexOf("Y"); //always true for part 2
        img.colorToCut = chooseColor.get(2); //for dangerous, the wire to cut is the 3rd laid down
        return img;
    }


    /**
     * Generate training images and store
     */
    private static void generateTrainingData() {
        for (int i = 0; i < numTrain; i++) {
            trainingImages.add(generateImage());
        }
    }

    /**
     * Generate testing images and store
     */
    private static void generateTestingData() {
        for (int i = 0; i < numTest; i++) {
            testingImages.add(generateImage());
        }
    }



    /**
     * Stochastic Gradient Descent --> one iteration/update
     */
    private static void trainModel() {
        //choose some random image from the data set
        Image img = trainingImages.get(rand(0, trainingImages.size()));
        calculateSoftmaxDenominator(img.image);

        //for each color, update corresponding weight vector based on how poorly model performs on img
        for (String color: colors.keySet()) {
            //f is the output of softmax, which is the probability of the current color being correct
            double f = softmax(color, img.image);
            //y is the actual output: 1 if this color
            int y = color.equals(img.colorToCut) ? 1 : 0;

            PixelVector[] subtractVector = scaleImage(alpha * (f - y), img.image);
            switch (color) {
                case "R":
                    weightVector_R = subtractVectors(weightVector_R, subtractVector);
                    break;
                case "G":
                    weightVector_G = subtractVectors(weightVector_G, subtractVector);
                    break;
                case "B":
                    weightVector_B = subtractVectors(weightVector_B, subtractVector);
                    break;
                case "Y":
                    weightVector_Y = subtractVectors(weightVector_Y, subtractVector);
                    break;
                default:
                    return;
            }
        }
    }

    /**
     * Generate the given initial weight vector
     */
    private static void generateInitialWeightVector(String color) {
        PixelVector[] w = new PixelVector[Main.D * Main.D + 1];
        for (int i = 0; i < w.length; i++) {
            w[i] = new PixelVector(initWeight, initWeight, initWeight, initWeight);
        }
        w[0] = new PixelVector(initWeight);

        switch (color) {
            case "R":
                weightVector_R = w;
                break;
            case "G":
                weightVector_G = w;
                break;
            case "B":
                weightVector_B = w;
                break;
            case "Y":
                weightVector_Y = w;
                break;
            default: return;
        }
    }

    /**
     * Driver method to run the experiment
     */
    public static void runExperiment() {
        trainingImages = new LinkedList<>();
        generateTrainingData();
        testingImages = new LinkedList<>();
        generateTestingData();

        generateInitialWeightVector("R");
        generateInitialWeightVector("G");
        generateInitialWeightVector("B");
        generateInitialWeightVector("Y");

        trainModel();

        double trainingLoss = calculateTrainingLoss();
        System.out.println("Initial Training Loss: " + trainingLoss + "\t\t\t");
        System.out.println("Initial Testing Loss: " + calculateTestingLoss());

        while (trainingLoss > trainingLossCutoff) {
            for (int i = 0; i < numTimesTrainAtOnce; i++) trainModel();
            trainingLoss = calculateTrainingLoss();
            System.out.println("Training Loss: " + trainingLoss + "\t\t\t" +
                    "Testing Loss: " + calculateTestingLoss());
        }
        System.out.println("********** terminate here to prevent overfitting **********");
        while (trainingLoss > divergenceCutoff) {
            for (int i = 0; i < numTimesTrainAtOnce; i++) trainModel();
            trainingLoss = calculateTrainingLoss();
            System.out.println("Training Loss: " + trainingLoss + "\t\t\t" +
                    "Testing Loss: " + calculateTestingLoss());
        }
    }
}
