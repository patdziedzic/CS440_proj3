public class Main {
    public static final int D = 20;

    /**
     * Main driver method to run task 1 and task 2
     */
    public static void main(String[] args) {
        //Task 1
        Task1.numTrain = 2000;
        Task1.numTest = 2000;
        Task1.trainingLossCutoff = 0.21;
        Task1.divergenceCutoff = 0.12;
        Task1.numTimesTrainAtOnce = 2000;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 2500;
        Task1.numTest = 2500;
        Task1.trainingLossCutoff = 0.19;
        Task1.divergenceCutoff = 0.14;
        Task1.numTimesTrainAtOnce = 4000;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 3000;
        Task1.numTest = 3000;
        Task1.trainingLossCutoff = 0.166;
        Task1.divergenceCutoff = 0.12;
        Task1.numTimesTrainAtOnce = 4000;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 5000;
        Task1.numTest = 5000;
        Task1.trainingLossCutoff = 0.205;
        Task1.divergenceCutoff = 0.14;
        Task1.numTimesTrainAtOnce = 8000;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");



        //Task 2
        Task1.numTrain = 5000;
        Task1.numTest = 5000;
        Task1.trainingLossCutoff = 0.2;
        Task1.divergenceCutoff = 0.14;
        Task1.numTimesTrainAtOnce = 4000;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 6000;
        Task1.numTest = 6000;
        Task1.trainingLossCutoff = 0.16;
        Task1.divergenceCutoff = 0.12;
        Task1.numTimesTrainAtOnce = 8000;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task2.numTrain = 7500;
        Task2.numTest = 7500;
        Task2.trainingLossCutoff = 0.1;
        Task2.divergenceCutoff = 0.08;
        Task2.numTimesTrainAtOnce = 8000;
        Task2.runExperiment();
        System.out.println("\n*******************************************************************************\n");

        Task1.numTrain = 10000;
        Task1.numTest = 10000;
        Task1.trainingLossCutoff = 0.075;
        Task1.divergenceCutoff = 0.05;
        Task1.numTimesTrainAtOnce = 12000;
        Task1.runExperiment();
        System.out.println("\n*******************************************************************************\n");
    }
}