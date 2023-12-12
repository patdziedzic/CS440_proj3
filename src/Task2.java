import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;

public class Task2 {
    public static double trainingLossCutoff; //to prevent overfitting
    public static int numTrain;
    private static LinkedList<Image> trainingImages = new LinkedList<>();
    public static int numTest;
    private static LinkedList<Image> testingImages = new LinkedList<>();
    private static final double alpha = 0.015;
    private static final double initWeight = 0.001;
    private static PixelVector[] weightVector = new PixelVector[Main.D*Main.D + 1];

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

    public static double sigmoid(double z) { return 1 / (1 + Math.exp(-z)); }

    /**
     * TRAINING loss
     */
    private static double calculateTrainingLoss() {
        double sum = 0.0;
        for (Image img : trainingImages) {
            int y = img.dangerous ? 1 : 0;
            sum += (-y * Math.log(sigmoid(dotProduct(img.image))))
                    -
                    ((1 - y) * Math.log(1 - sigmoid(dotProduct(img.image))));
        }

        return ((double) 1 / trainingImages.size()) * sum;
    }

    /**
     * TESTING loss
     */
    private static double calculateTestingLoss() {
        double sum = 0.0;
        for (Image img : testingImages) {
            int y = img.dangerous ? 1 : 0;
            sum += (-y * Math.log(sigmoid(dotProduct(img.image))))
                    -
                    ((1 - y) * Math.log(1 - sigmoid(dotProduct(img.image))));
        }

        return ((double) 1 / testingImages.size()) * sum;
    }

    /**
     * Dot product of weight with given image vector
     */
    public static double dotProduct(PixelVector[] image) {
        double sum = 0;
        //go through all vectors
        for (int i = 0; i < image.length; i++) {
            //go through all bits in the vector
            for (int j = 0; j < image[i].vector.length; j++)
                sum += weightVector[i].vector[j] * image[i].vector[j];
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
        img.dangerous = chooseColor.indexOf("R") < chooseColor.indexOf("Y");
        img.colorToCut = chooseColor.get(2); //for dangerous, the wire to cut is the 3rd laid down
        return img;
    }


    private static void generateTrainingData() {
        for (int i = 0; i < numTrain; i++) {
            trainingImages.add(generateImage());
        }
    }

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

        //update weightVector based on how poorly model performs on img
        PixelVector[] newWeightVector = new PixelVector[weightVector.length];
        for (int i = 0; i < newWeightVector.length; i++) {
            newWeightVector[i] = new PixelVector();
        }

        double f = sigmoid(dotProduct(img.image));
        int y = img.dangerous ? 1 : 0;

        PixelVector[] subtractVector = scaleImage(alpha*(f-y), img.image);

        //go through all vectors
        for (int i = 0; i < newWeightVector.length; i++) {
            newWeightVector[i].vector = new double[weightVector[i].vector.length];
            //go through all bits in the vector
            for (int j = 0; j < newWeightVector[i].vector.length; j++)
                newWeightVector[i].vector[j] = weightVector[i].vector[j] - subtractVector[i].vector[j];
        }
        weightVector = newWeightVector;
    }

    public static void runExperiment() {
        trainingImages = new LinkedList<>();
        generateTrainingData();
        testingImages = new LinkedList<>();
        generateTestingData();

        //generate initial weight vector
        weightVector = new PixelVector[Main.D*Main.D + 1];
        for (int i = 0; i < weightVector.length; i++) {
            weightVector[i] = new PixelVector(initWeight, initWeight, initWeight, initWeight);
        }
        weightVector[0] = new PixelVector(initWeight);

        trainModel();

        double trainingLoss = calculateTrainingLoss();
        System.out.println("Initial Training Loss: " + trainingLoss + "\t\t\t");
        System.out.println("Initial Testing Loss: " + calculateTestingLoss());

        while (trainingLoss > trainingLossCutoff) {
            for (int i = 0; i < 2000; i++) trainModel();
            trainingLoss = calculateTrainingLoss();
            System.out.println("Training Loss: " + trainingLoss + "\t\t\t" +
                    "Testing Loss: " + calculateTestingLoss());
        }
    }
}
