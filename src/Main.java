import java.util.*;

public class Main {
    public static final int D = 20;
    private static int numTrain = 2000;
    private static LinkedList<Image> trainingImages = new LinkedList<>();
    private static int numTest = 2000;
    private static LinkedList<Image> testingImages = new LinkedList<>();
    private static final double alpha = 0.25;
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

    public static double loss() {
        double sum = 0.0;
        for (Image img : trainingImages) {
            int y = img.dangerous ? 1 : 0;
            sum += (-y * Math.log(sigmoid(dotProduct(weightVector, img.image))))
                    -
                    ((1 - y) * Math.log(1 - sigmoid(dotProduct(weightVector, img.image))));
        }

        return ((double) 1 / trainingImages.size()) * sum;
    }

    /**
     * Dot product of 2 images as PixelVector[] arrays
     */
    public static double dotProduct(PixelVector[] w, PixelVector[] x) {
        double sum = 0;
        //go through all vectors
        for (int i = 0; i < x.length; i++) {
            //go through all bits in the vector
            for (int j = 0; j < x[i].vector.length; j++)
                sum += w[i].vector[j] * x[i].vector[j];
        }
        return sum;
    }

    /**
     * Scalar multiplication for an Image
     */
    public static PixelVector[] scaleImage(double scalar, PixelVector[] img) {
        PixelVector[] resultImg = new PixelVector[img.length];
        Arrays.fill(resultImg, new PixelVector());
        //go through all vectors
        for (int i = 0; i < img.length; i++) {
            resultImg[i].vector = img[i].vector;
            //go through all bits in the vector
            for (int j = 0; j < img[i].vector.length; j++)
                resultImg[i].vector[j] = scalar * img[i].vector[j];
        }
        return resultImg;
    }

    /*private static void runTests(int bot) {
        LinkedList<Integer> testResults = new LinkedList<>();
        for (int test = 1; test <= numTests; test++) {
            ship = Ship.makeShip();
            openCells = new ArrayList<>();
            for (int i = 0; i < Ship.D; i++){
                for (int j = 0; j < Ship.D; j++){
                    if (ship[i][j].isOpen)
                        openCells.add(ship[i][j]);
                }
            }
            numActions = 0;
            switch (bot) {
                case 1 -> DeterministicBots.runBot1();
                case 2 -> DeterministicBots.runBot2();
                case 3 -> ProbabilisticBots.runBot3();
                case 4 -> ProbabilisticBots.runBot4();
                case 5 -> MultipleLeaksBots.runBot5();
                case 6 -> MultipleLeaksBots.runBot6();
                case 7 -> MultipleLeaksBots.runBot7();
                case 8 -> MultipleLeaksBots.runBot8();
                case 9 -> MultipleLeaksBots.runBot9();
                default -> numActions = 0;
            }

            //Ship.printShip(ship);

            if (numActions == null) //if null, forget this test
                test--;
            else
                testResults.add(numActions);
            //System.out.println("Test " + test + " completed.");
        }

        int totalActions = 0;
        for (Integer result : testResults) {
            if (result != null)
                totalActions += result;
        }
        double avg = totalActions / (double) numTests;

        switch (bot) {
            case 1, 2, 5, 6 ->
                    System.out.println("Avg Actions Taken for k = " + k + " is " + avg);
            case 3, 4, 7, 8, 9 ->
                    System.out.println("Avg Actions Taken for alpha = " + alpha + " is " + avg);
            default -> System.out.println("Bot number out of range.");
        }

    }*/

    public static Image generateImage() {
        PixelVector[] image = new PixelVector[D*D + 1];
        Arrays.fill(image, new PixelVector());
        image[0] = new PixelVector(1);

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

        //randomly pick either to start with row or col
        if (Math.random() < 0.5) {
            //do row first
            for (int c = 0; c < D; c++)
                image[randRow1*D + 1 + c] = colors.get(chooseColor.get(0));
            for (int r = 0; r < D*D; r += D)
                image[randCol1 + 1 + r] = colors.get(chooseColor.get(1));

            for (int c = 0; c < D; c++)
                image[randRow2 + 1 + c] = colors.get(chooseColor.get(2));
            for (int r = 0; r < D*D; r += D)
                image[randCol2 + 1 + r] = colors.get(chooseColor.get(3));
        }
        else {
            //do col first
            for (int r = 0; r < D*D; r += D)
                image[randCol1 + 1 + r] = colors.get(chooseColor.get(0));
            for (int c = 0; c < D; c++)
                image[randRow1*D + 1 + c] = colors.get(chooseColor.get(1));

            for (int r = 0; r < D*D; r += D)
                image[randCol2 + 1 + r] = colors.get(chooseColor.get(2));
            for (int c = 0; c < D; c++)
                image[randRow2 + 1 + c] = colors.get(chooseColor.get(3));
        }
        Image img = new Image();
        img.image = image;
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
        //w(k+1) = w(k) - alpha * (f - y)*x
        PixelVector[] newWeightVector = new PixelVector[D*D + 1];
        Arrays.fill(newWeightVector, new PixelVector());

        double f = sigmoid(dotProduct(weightVector, img.image));
        int y = img.dangerous ? 1 : 0;

        PixelVector[] subtractVector = scaleImage(alpha*(f-y), img.image);

        //go through all vectors
        for (int i = 0; i < newWeightVector.length; i++) {
            newWeightVector[i].vector = weightVector[i].vector;
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
        numTrain = 2000;
        generateTrainingData();
        numTest = 2000;
        generateTestingData();

        Arrays.fill(weightVector, new PixelVector(initWeight, initWeight, initWeight, initWeight));
        weightVector[0] = new PixelVector(initWeight);

        trainModel();
        System.out.println("Init Loss: " + loss());

        //for (int i = 0; i < 10; i++) {
        while (true) {
            for (int i = 0; i < 10; i++) trainModel();
            System.out.println("Loss: " + loss());
        }
        //}

        /*
        Image img = trainingImages.get(rand(0, trainingImages.size()));
        for (PixelVector pv : img.image)
            System.out.println(Arrays.toString(pv.vector));*/
    }




}