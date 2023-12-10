import java.util.*;

public class Main {
    public static final int D = 20;
    private static int numTrain = 2000;
    private static LinkedList<Image> trainingImages = new LinkedList<>();
    private static int numTest = 2000;
    private static LinkedList<Image> testingImages = new LinkedList<>();
    private static final double alpha = 0.02; //0.01
    private static final double initWeight = 0.001;
    private static PixelVector[] weightVector = new PixelVector[D*D + 1];

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
    public static double calculateTrainingLoss() {
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
    public static double calculateTestingLoss() {
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
    public static PixelVector[] scaleImage(double scalar, PixelVector[] img) {
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

    public static Image generateImage() {
        PixelVector[] pvArray = new PixelVector[D*D + 1];
        for (int i = 0; i < pvArray.length; i++) {
            pvArray[i] = new PixelVector();
        }
        pvArray[0] = new PixelVector(1);

        ArrayList<String> chooseColor = new ArrayList<>();
        chooseColor.add("R");
        chooseColor.add("G");
        chooseColor.add("B");
        chooseColor.add("Y");
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
            for (int c = 0; c < D; c++)
                pvArray[randRow1*D + 1 + c] = new PixelVector(color0);
            for (int r = 0; r < D*D; r += D)
                pvArray[randCol1 + 1 + r] = new PixelVector(color1);

            for (int c = 0; c < D; c++)
                pvArray[randRow2 + 1 + c] = new PixelVector(color2);
            for (int r = 0; r < D*D; r += D)
                pvArray[randCol2 + 1 + r] = new PixelVector(color3);
        }
        else {
            //do col first
            for (int r = 0; r < D*D; r += D)
                pvArray[randCol1 + 1 + r] = new PixelVector(color0);
            for (int c = 0; c < D; c++)
                pvArray[randRow1*D + 1 + c] = new PixelVector(color1);

            for (int r = 0; r < D*D; r += D)
                pvArray[randCol2 + 1 + r] = new PixelVector(color2);
            for (int c = 0; c < D; c++)
                pvArray[randRow2 + 1 + c] = new PixelVector(color3);
        }
        Image img = new Image();
        img.image = pvArray;
        img.dangerous = chooseColor.indexOf("R") < chooseColor.indexOf("Y");
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

    /**
     * Main driver method to run the tests for each bot
     */
    public static void main(String[] args) {
        System.out.println("These are the colors' addresses:\n");
        System.out.println(colors.get("R"));
        System.out.println(colors.get("G"));
        System.out.println(colors.get("B"));
        System.out.println(colors.get("Y"));
        System.out.println("\n\n\n\n\n");

        numTrain = 5000;
        generateTrainingData();
        numTest = 5000;
        generateTestingData();

        //generate initial weight vector
        for (int i = 0; i < weightVector.length; i++) {
            weightVector[i] = new PixelVector(initWeight, initWeight, initWeight, initWeight);
        }
        weightVector[0] = new PixelVector(initWeight);

        trainModel();

        System.out.println("Initial Training Loss: " + calculateTrainingLoss());
        System.out.println("Initial Testing Loss: " + calculateTestingLoss());

        //for (int i = 0; i < 10; i++) {
        while (true) {
            for (int i = 0; i < 2000; i++) trainModel();
            System.out.println("Training Loss: " + calculateTrainingLoss() + "          " +
                    "Testing Loss: " + calculateTestingLoss());

            /*Image img = trainingImages.get(rand(0, trainingImages.size()));
            double f = sigmoid(dotProduct(img.image));
            int y = img.dangerous ? 1 : 0;
            PixelVector[] gradient = scaleImage((f-y), img.image);
            System.out.println("GRADIENT:");
            for (PixelVector pv : gradient)
                System.out.println(Arrays.toString(pv.vector));
            System.out.println("\n\n");*/
        }
        //}

        /*

        Image img = trainingImages.get(rand(0, trainingImages.size()));
        //for (PixelVector pv : img.image)
            //System.out.println(pv);

        //System.out.println("\n\n\n");

        for (PixelVector pv : img.weightVector)
            System.out.println(Arrays.toString(pv.vector));

         */
    }




}