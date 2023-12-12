public class Main {
    public static final int D = 20;

    /**
     * Main driver method to run task 1 and task 2
     */
    public static void main(String[] args) {
        //Task 1
        Task1.numTrain = 2000;
        Task1.numTest = 2000;
        Task1.trainingLossCutoff = 0.24;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 2500;
        Task1.numTest = 2500;
        Task1.trainingLossCutoff = 0.193;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 3000;
        Task1.numTest = 3000;
        Task1.trainingLossCutoff = 0.166;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 5000;
        Task1.numTest = 5000;
        Task1.trainingLossCutoff = 0.205;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");



        //Task 2

    }
}