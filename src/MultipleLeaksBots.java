import java.util.HashMap;
import java.util.LinkedList;

/**
 * Class for Multiple Leaks Bots (5-9)
 */
public class MultipleLeaksBots extends DeterministicBots {
    private static HashMap<Cell, HashMap<Cell, Double>> pairings = new HashMap<>();

    /*
     * Run an experiment on Bot 5
     */
    public static void runBot5() {
        //initialize the bot
        int randIndex = rand(0, openCells.size()-1);
        Cell bot = openCells.get(randIndex);
        bot.isBot = true;
        LinkedList<Cell> detSquare = getDetectionSquare(bot);

        //initialize leak1
        Cell leak1 = null;
        //the leak must initially be placed in a random open cell outside of the detection square
        while (leak1 == null) {
            randIndex = rand(0, openCells.size() - 1);
            Cell tempLeak = openCells.get(randIndex);

            boolean found = false;
            for (Cell cell : detSquare) {
                if (cell.equals(tempLeak)) {
                    found = true;
                    break;
                }
            }
            if (!found) leak1 = tempLeak;
        }
        leak1.isLeak = true;
        
        //initialize leak2
        Cell leak2 = null;
        //the leak must initially be placed in a random open cell outside of the detection square
        while (leak2 == null) {
            randIndex = rand(0, openCells.size() - 1);
            Cell tempLeak = openCells.get(randIndex);
            if (!tempLeak.isLeak) {
                boolean found = false;
                for (Cell cell : detSquare) {
                    if (cell.equals(tempLeak)) {
                        found = true;
                        break;
                    }
                }
                if (!found) leak2 = tempLeak;
            }
        }
        leak2.isLeak = true;

        bot.noLeak = true;
        for (Cell cell : detSquare) cell.noLeak = true;

        while (!bot.isLeak && (leak1.isLeak || leak2.isLeak)) {
            //BFS Shortest Path from bot -> nearest potential leak
            LinkedList<Cell> shortestPath = Bfs.detSP_BFS(bot);
            if (shortestPath == null) {
                numActions = null;
                return;
            }
            shortestPath.removeFirst();

            //move the bot to the nearest potential leak
            bot = moveBot(bot, shortestPath);
            //if both leaks have not been plugged
            if (leak1.isLeak && leak2.isLeak) {
                //if bot reached one of the leaks
                if (bot.isLeak) bot.isLeak = false;
                bot.noLeak = true;
                detSenseAction_MultipleLeaks(bot, leak1, leak2);
            }
            else {
                //if bot reached one of the leaks and the other has been plugged
                if (bot.isLeak && (leak1.isLeak ^ leak2.isLeak)) return;
                bot.noLeak = true;
                detSenseAction(bot);
            }
        }
    }

    /**
     * Deterministic Sense Action for Multiple Leaks (update cells accordingly)
     */
    public static void detSenseAction_MultipleLeaks(Cell bot, Cell leak1, Cell leak2) {
        LinkedList<Cell> detSquare = getDetectionSquare(bot);
        numActions++;
        if (!leakInSquare(detSquare)) {
            //set noLeak = true for everything in the square
            for (Cell cell : detSquare) cell.noLeak = true;
        }
        else {
            //leak detected --> look only at the cells within the detection square
            while (!bot.isLeak) {
                //BFS Shortest Path from bot -> nearest potential leak in square
                LinkedList<Cell> shortestPath = Bfs.detSP_BFS(bot, detSquare);
                if (shortestPath == null) return;
                shortestPath.removeFirst();

                //move the bot to the nearest potential leak
                bot = moveBot(bot, shortestPath);
                //if bot reached one of the leaks and both have not been plugged
                if (bot.isLeak && leak1.isLeak && leak2.isLeak) {
                    bot.isLeak = false;
                    bot.noLeak = true;
                    return;
                }
                //else if bot reached one of the leaks and the other has been plugged
                else if (bot.isLeak && (leak1.isLeak ^ leak2.isLeak)) return;
                else bot.noLeak = true;
            }
        }
    }

    /**
     * Run an experiment for Bot 6
     */
    public static void runBot6() {
        //initialize the bot
        int randIndex = rand(0, openCells.size()-1);
        Cell bot = openCells.get(randIndex);
        bot.isBot = true;
        LinkedList<Cell> detSquare = getDetectionSquare(bot);

        //initialize leak1
        Cell leak1 = null;
        //the leak must initially be placed in a random open cell outside of the detection square
        while (leak1 == null) {
            randIndex = rand(0, openCells.size() - 1);
            Cell tempLeak = openCells.get(randIndex);

            boolean found = false;
            for (Cell cell : detSquare) {
                if (cell.equals(tempLeak)) {
                    found = true;
                    break;
                }
            }
            if (!found) leak1 = tempLeak;
        }
        leak1.isLeak = true;
        
        //initialize leak2
        Cell leak2 = null;
        //the leak must initially be placed in a random open cell outside of the detection square
        while (leak2 == null) {
            randIndex = rand(0, openCells.size() - 1);
            Cell tempLeak = openCells.get(randIndex);
            if (!tempLeak.isLeak) {
                boolean found = false;
                for (Cell cell : detSquare) {
                    if (cell.equals(tempLeak)) {
                        found = true;
                        break;
                    }
                }
                if (!found) leak2 = tempLeak;
            }
        }
        leak2.isLeak = true;

        bot.noLeak = true;
        for (Cell cell : detSquare) cell.noLeak = true;

        while (!bot.isLeak && (leak1.isLeak || leak2.isLeak)) {
            //BFS Shortest Path from bot -> nearest potential leak
            LinkedList<Cell> shortestPath = Bfs.detSP_BFS(bot);
            if (shortestPath == null) {
                numActions = null;
                return;
            }
            shortestPath.removeFirst();

            //move the bot one step to the nearest potential leak
            Cell neighbor = shortestPath.removeFirst();
            bot.isBot = false;
            neighbor.isBot = true;
            bot = neighbor;
            numActions++;

            //if both leaks have not been plugged
            if (leak1.isLeak && leak2.isLeak) {
                //if bot reached one of the leaks
                if (bot.isLeak) bot.isLeak = false;
                bot.noLeak = true;
                detSenseAction_MultipleLeaks(bot, leak1, leak2);
            }
            else {
                //if bot reached one of the leaks and the other has been plugged
                if (bot.isLeak && (leak1.isLeak ^ leak2.isLeak)) return;
                bot.noLeak = true;
                detSenseAction(bot);
            }
        }
    }


    /**
     * Run an experiment for Bot 7
     * Modeled after Bot 3; admittedly incorrect probabilities
     */
    public static void runBot7() {
        //initialize the bot
        int randIndex = rand(0, openCells.size()-1);
        Cell bot = openCells.get(randIndex);
        bot.isBot = true;

        //initialize leak1
        randIndex = rand(0, openCells.size()-1);
        Cell leak1 = openCells.get(randIndex);
        leak1.isLeak = true;
        //if the leak spawns on the bot, ignore test case
        if (bot.isLeak) {
            numActions = null;
            return;
        }

        //initialize leak2
        Cell leak2 = null;
        while (leak2 == null) {
            randIndex = rand(0, openCells.size() - 1);
            Cell tempLeak = openCells.get(randIndex);
            if (!tempLeak.isLeak) leak2 = tempLeak;
        }
        leak2.isLeak = true;
        //if the leak spawns on the bot, ignore test case
        if (bot.isLeak) {
            numActions = null;
            return;
        }

        bot.noLeak = true;

        //set initial probabilities and get the cell with the max
        int size = openCells.size();
        double maxProb = 1/(double)size;
        for (Cell cell : openCells) {
            cell.setProbLeak(maxProb);
        }
        bot.setProbLeak(0);
        Cell maxProbCell;

        while (!bot.isLeak && (leak1.isLeak || leak2.isLeak)) {
            //BFS Shortest Path from bot -> cell with highest P(L)
            maxProb = 0.0;
            maxProbCell = null;
            for (Cell cell : openCells) {
                if (maxProb < cell.getProbLeak()) {
                    maxProb = cell.getProbLeak();
                    maxProbCell = cell;
                }
            }
            LinkedList<Cell> shortestPath = Bfs.SP_BFS(bot, maxProbCell);
            if (shortestPath == null) {
                numActions = null;
                return;
            }
            shortestPath.removeFirst();

            //move the bot one step toward cell with highest P(L)
            Cell neighbor = shortestPath.removeFirst();
            bot.isBot = false;
            neighbor.isBot = true;
            bot = neighbor;
            numActions++;
            Bfs.updateDistances(bot);

            //if bot reached one of the leaks and both have not been plugged
            if (bot.isLeak && leak1.isLeak && leak2.isLeak) {
                bot.isLeak = false;
                bot.noLeak = true;
            }
            //else if bot reached one of the leaks and the other has been plugged
            else if (bot.isLeak && (leak1.isLeak ^ leak2.isLeak)) return;
            else bot.noLeak = true;

            updateProb_Step(bot);

            //Sense Action
            if (bot.equals(maxProbCell)) {
                if (leak1.isLeak && leak2.isLeak)
                    probSenseAction_Bot7(bot, leak1, leak2);
                else if (leak1.isLeak)
                    probSenseAction(bot, leak1);
                else
                    probSenseAction(bot, leak2);
            }
        }
    }

    /**
     * Update P(L) for each cell after taking a step
     */
    private static void updateProb_Step(Cell bot) {
        //update P(L) for each cell j to be P(leak in cell j | leak is not in bot cell)
        // --> = P(leak in cell j) / [1 - P(leak is in bot cell)]
        double denominator = 1 - bot.getProbLeak();

        //calculate and store the new values
        double[][] newValues = new double[Ship.D][Ship.D];
        for (Cell cell : openCells) {
            newValues[cell.getRow()][cell.getCol()] = cell.getProbLeak() / denominator;
        }

        //copy over the new values to the cells
        for (Cell cell : openCells) {
            cell.setProbLeak(newValues[cell.getRow()][cell.getCol()]);
        }

        bot.setProbLeak(0);
    }

    /**
     * Update P(L) for each cell after a sense action
     */
    private static void updateProb_Sense(double probB) {
        //calculate the denominator
        double denominator = 0.0;
        for (Cell cell : openCells) {
            denominator += cell.getProbLeak() * probB;
        }

        //calculate and store the new values
        double[][] newValues = new double[Ship.D][Ship.D];
        for (Cell cell : openCells) {
            newValues[cell.getRow()][cell.getCol()] = (cell.getProbLeak() * probB) / denominator;
        }

        //copy over the new values to the cells
        for (Cell cell : openCells) {
            cell.setProbLeak(newValues[cell.getRow()][cell.getCol()]);
        }
    }

    /**
     * Sense the leak and update based on whether a beep was heard
     */
    private static void probSenseAction(Cell bot, Cell leak) {
        //sense action
        bot.setBeepProb(leak);
        boolean beep = Math.random() <= bot.getBeepProb(); //true if beep occurred
        numActions++;

        //update P(L) for each cell j to be P(Lj | B/~B in i)
        // --> = P(Lj)*P(B/~B in i | Lj) / summation{P(Lj')*P(B/~B in i | Lj')}
        if (beep) {
            updateProb_Sense(bot.getBeepProb());
        }
        else {
            updateProb_Sense(1 - bot.getBeepProb());
        }
    }

    /**
     * Sense the leak and update based on whether a beep was heard given multiple leaks
     */
    private static void probSenseAction_Bot7(Cell bot, Cell leak1, Cell leak2) {
        //sense action
        bot.setBeepProb(leak1);
        boolean beep1 = Math.random() <= bot.getBeepProb(); //true if beep occurred
        bot.setBeepProb(leak2);
        boolean beep2 = Math.random() <= bot.getBeepProb(); //true if beep occurred
        numActions++;

        //update P(L) for each cell j to be P(Lj | B/~B in i)
        // --> = P(Lj)*P(B/~B in i | Lj) / summation{P(Lj')*P(B/~B in i | Lj')}
        if (beep1 || beep2) { // if either leak caused a beep
            updateProb_Sense(bot.getBeepProb());
        }
        else {
            updateProb_Sense(1 - bot.getBeepProb());
        }
    }



    /**
     * Run an experiment for Bot 8
     * Modeled after Bot 7 but adjusted probability calculations
     */
    public static void runBot8() {
        //initialize the bot
        int randIndex = rand(0, openCells.size()-1);
        Cell bot = openCells.get(randIndex);
        bot.isBot = true;

        //initialize leak1
        randIndex = rand(0, openCells.size()-1);
        Cell leak1 = openCells.get(randIndex);
        leak1.isLeak = true;
        //if the leak spawns on the bot, ignore test case
        if (bot.isLeak) {
            numActions = null;
            return;
        }

        //initialize leak2
        Cell leak2 = null;
        while (leak2 == null) {
            randIndex = rand(0, openCells.size() - 1);
            Cell tempLeak = openCells.get(randIndex);
            if (!tempLeak.isLeak) leak2 = tempLeak;
        }
        leak2.isLeak = true;
        //if the leak spawns on the bot, ignore test case
        if (bot.isLeak) {
            numActions = null;
            return;
        }

        bot.noLeak = true;

        //set initial probabilities to 2/(n * (n-1)) and randomly pick a max
        int n = openCells.size() - 1; //(-1) to account for the bot initially not being the leak
        double maxProbLeakPair = 2 / (double) (n * (n-1));

        double sumProbLeak = 0.0;

        for (int i = 0; i < openCells.size(); i++) {
            Cell c1 = openCells.get(i);
            if (!c1.equals(bot)) {
                HashMap<Cell, Double> pairsForGivenCell = new HashMap<>();
                //for all cells before i, just use the previously calculated value
                for (int j = 0; j < i; j++) {
                    Cell c2 = openCells.get(j);
                    if (!c1.equals(c2) && !c2.equals(bot) && !c1.noLeak && !c2.noLeak) {
                        pairsForGivenCell.put(c2, pairings.get(c2).get(c1));
                        sumProbLeak += pairings.get(c2).get(c1);
                    }
                }
                //for all cells after i, set the initial probability for the pair
                for (int j = i + 1; j < openCells.size(); j++) {
                    Cell c2 = openCells.get(j);
                    if (!c1.equals(c2) && !c2.equals(bot) && !c1.noLeak && !c2.noLeak) {
                        pairsForGivenCell.put(c2, maxProbLeakPair);
                        sumProbLeak += maxProbLeakPair;
                    }
                }
                pairings.put(c1, pairsForGivenCell);
                //set P(L) for c1 to be the summation of all the pairings for c1
                c1.setProbLeak(sumProbLeak);
                sumProbLeak = 0.0;
            }
        }

        bot.setProbLeak(0);
        double maxProb;
        Cell maxProbCell;

        while (!bot.isLeak && leak1.isLeak && leak2.isLeak) {
            //BFS Shortest Path from bot -> cell with highest P(L)
            maxProb = 0.0;
            maxProbCell = null;
            for (Cell cell : openCells) {
                if (maxProb < cell.getProbLeak()) {
                    maxProb = cell.getProbLeak();
                    maxProbCell = cell;
                }
            }
            LinkedList<Cell> shortestPath = Bfs.SP_BFS(bot, maxProbCell);
            if (shortestPath == null) {
                numActions = null;
                return;
            }
            shortestPath.removeFirst();

            //move the bot one step toward cell with highest P(L)
            Cell neighbor = shortestPath.removeFirst();
            bot.isBot = false;
            neighbor.isBot = true;
            bot = neighbor;
            numActions++;
            Bfs.updateDistances(bot);

            //if bot reached one of the leaks and both have not been plugged
            if (bot.isLeak && leak1.isLeak && leak2.isLeak)
                bot.isLeak = false;

            bot.noLeak = true;
            updateProb_Step_Bot8(bot);

            //Sense Action
            if (bot.equals(maxProbCell)) {
                if (leak1.isLeak && leak2.isLeak)
                    probSenseAction_Bot8(bot, leak1, leak2);
                else if (leak1.isLeak)
                    probSenseAction(bot, leak1);
                else
                    probSenseAction(bot, leak2);
            }
        }

        Cell leak;
        if (leak1.isLeak) leak = leak1;
        else leak = leak2;

        //run bot 3 algorithm
        while (!bot.isLeak) {
            //BFS Shortest Path from bot -> cell with highest P(L)
            maxProb = 0.0;
            maxProbCell = null;
            for (Cell cell : openCells) {
                if (maxProb < cell.getProbLeak()) {
                    maxProb = cell.getProbLeak();
                    maxProbCell = cell;
                }
            }
            LinkedList<Cell> shortestPath = Bfs.SP_BFS(bot, maxProbCell);
            if (shortestPath == null) {
                numActions = null;
                return;
            }
            shortestPath.removeFirst();

            //move the bot one step toward cell with highest P(L)
            Cell neighbor = shortestPath.removeFirst();
            bot.isBot = false;
            neighbor.isBot = true;
            bot = neighbor;
            numActions++;
            Bfs.updateDistances(bot);

            if (bot.isLeak) return;

            bot.noLeak = true;
            updateProb_Step(bot);

            //Sense Action
            if (bot.equals(maxProbCell))
                probSenseAction(bot, leak);
        }
    }

    /**
     * Update P(Lj, Lk) for each pairing after taking a step
     */
    private static void updateProb_Step_Bot8(Cell bot) {
        //update P(Lj, Lk) for each pairing to be P(Lj, Lk) / [1 - P(Li)]
        double denominator = 1 - bot.getProbLeak();

        HashMap<Cell, HashMap<Cell, Double>> newPairings = new HashMap<>();
        double sumProbLeak = 0.0;
        for (int i = 0; i < openCells.size(); i++) {
            Cell c1 = openCells.get(i);
            if (!c1.equals(bot)) {
                HashMap<Cell, Double> pairsForGivenCell = new HashMap<>();
                //for all cells before i, just use the previously calculated value
                for (int j = 0; j < i; j++) {
                    Cell c2 = openCells.get(j);
                    if (!c2.equals(bot) && !c1.equals(c2) && !c1.noLeak && !c2.noLeak) {
                        pairsForGivenCell.put(c2, pairings.get(c2).get(c1));
                        sumProbLeak += pairings.get(c2).get(c1);
                    }
                }
                //for all cells after i, calculate new probability
                for (int j = i + 1; j < openCells.size(); j++) {
                    Cell c2 = openCells.get(j);
                    if (!c2.equals(bot) && !c1.equals(c2) && !c1.noLeak && !c2.noLeak) {
                        //divide by denominator for each pairing
                        double newProb = pairings.get(c1).get(c2) / denominator;
                        pairsForGivenCell.put(c2, newProb);
                        sumProbLeak += newProb;
                    }
                }
                newPairings.put(c1, pairsForGivenCell);
                //set P(L) for c1 to be the summation of all the pairings for c1
                c1.setProbLeak(sumProbLeak);
                sumProbLeak = 0.0;
            }
        }
        pairings = newPairings;

        bot.setProbLeak(0);
    }

    /**
     * Update P(Lj, Lk) for each pairing after a sense action
     */
    private static void updateProb_Sense_Bot8(double probB) {
        //calculate the denominator
        double denominator = 0.0;
        for (int i = 0; i < openCells.size(); i++) {
            Cell c1 = openCells.get(i);
            //start from i+1 to count each pair only once when computing denominator
            for (int j = i + 1; j < openCells.size(); j++) {
                Cell c2 = openCells.get(j);
                if (!c1.equals(c2) && !c1.noLeak && !c2.noLeak)
                    denominator += pairings.get(c1).get(c2) * probB;
            }
        }

        //calculate and store the new values
        HashMap<Cell, HashMap<Cell, Double>> newPairings = new HashMap<>();
        double sumProbLeak = 0.0;
        for (int i = 0; i < openCells.size(); i++) {
            Cell c1 = openCells.get(i);
            HashMap<Cell, Double> pairsForGivenCell = new HashMap<>();
            //for all cells before i, just use the previously calculated value
            for (int j = 0; j < i; j++) {
                Cell c2 = openCells.get(j);
                if (!c1.equals(c2) && !c1.noLeak && !c2.noLeak) {
                    pairsForGivenCell.put(c2, pairings.get(c2).get(c1));
                    sumProbLeak += pairings.get(c2).get(c1);
                }
            }
            //for all cells after i, calculate new probability
            for (int j = i + 1; j < openCells.size(); j++) {
                Cell c2 = openCells.get(j);
                if (!c1.equals(c2) && !c1.noLeak && !c2.noLeak) {
                    double newProb = (pairings.get(c1).get(c2) * probB) / denominator;
                    pairsForGivenCell.put(c2, newProb);
                    sumProbLeak += newProb;
                }
            }
            newPairings.put(c1, pairsForGivenCell);
            //set P(L) for c1 to be the summation of all the pairings for c1
            c1.setProbLeak(sumProbLeak);
            sumProbLeak = 0.0;
        }
        pairings = newPairings;
    }

    /**
     * Sense the leak and update based on whether a beep was heard given multiple leaks
     */
    private static void probSenseAction_Bot8(Cell bot, Cell leak1, Cell leak2) {
        //sense action
        bot.setBeepProb(leak1);
        double prob1 = bot.getBeepProb();
        boolean beep1 = Math.random() <= bot.getBeepProb(); //true if beep occurred

        bot.setBeepProb(leak2);
        double prob2 = bot.getBeepProb();
        boolean beep2 = Math.random() <= bot.getBeepProb(); //true if beep occurred

        numActions++;

        //update P(Lj, Lk) for each cell based on if the sensor beeped or didn't beep
        if (beep1 || beep2) { // if either leak caused a beep
            //P(B in i | L in j and L in k)
            bot.setBeepProb(1 - ( (1-prob1) * (1-prob2) ));
            updateProb_Sense_Bot8(bot.getBeepProb());
        }
        else {
            //P(~B in i | L in j and L in k)
            bot.setBeepProb((1-prob1) * (1-prob2));
            updateProb_Sense_Bot8(1 - bot.getBeepProb());
        }
    }



    /**
     * Run an experiment for Bot 9
     * Modeled after Bot 8 and Bot 4
     */
    public static void runBot9() {
        //initialize the bot
        int randIndex = rand(0, openCells.size()-1);
        Cell bot = openCells.get(randIndex);
        bot.isBot = true;

        //initialize leak1
        randIndex = rand(0, openCells.size()-1);
        Cell leak1 = openCells.get(randIndex);
        leak1.isLeak = true;
        //if the leak spawns on the bot, ignore test case
        if (bot.isLeak) {
            numActions = null;
            return;
        }

        //initialize leak2
        Cell leak2 = null;
        while (leak2 == null) {
            randIndex = rand(0, openCells.size() - 1);
            Cell tempLeak = openCells.get(randIndex);
            if (!tempLeak.isLeak) leak2 = tempLeak;
        }
        leak2.isLeak = true;
        //if the leak spawns on the bot, ignore test case
        if (bot.isLeak) {
            numActions = null;
            return;
        }

        bot.noLeak = true;

        //set initial probabilities to 2/(n * (n-1)) and randomly pick a max
        int n = openCells.size() - 1; //(-1) to account for the bot initially not being the leak
        double maxProbLeakPair = 2 / (double) (n * (n-1));

        double sumProbLeak = 0.0;

        for (int i = 0; i < openCells.size(); i++) {
            Cell c1 = openCells.get(i);
            if (!c1.equals(bot)) {
                HashMap<Cell, Double> pairsForGivenCell = new HashMap<>();
                //for all cells before i, just use the previously calculated value
                for (int j = 0; j < i; j++) {
                    Cell c2 = openCells.get(j);
                    if (!c1.equals(c2) && !c2.equals(bot) && !c1.noLeak && !c2.noLeak) {
                        pairsForGivenCell.put(c2, pairings.get(c2).get(c1));
                        sumProbLeak += pairings.get(c2).get(c1);
                    }
                }
                //for all cells after i, set the initial probability for the pair
                for (int j = i + 1; j < openCells.size(); j++) {
                    Cell c2 = openCells.get(j);
                    if (!c1.equals(c2) && !c2.equals(bot) && !c1.noLeak && !c2.noLeak) {
                        pairsForGivenCell.put(c2, maxProbLeakPair);
                        sumProbLeak += maxProbLeakPair;
                    }
                }
                pairings.put(c1, pairsForGivenCell);
                //set P(L) for c1 to be the summation of all the pairings for c1
                c1.setProbLeak(sumProbLeak);
                sumProbLeak = 0.0;
            }
        }

        bot.setProbLeak(0);
        double maxProb;
        Cell maxProbCell;
        boolean performSense = true;

        while (!bot.isLeak && leak1.isLeak && leak2.isLeak) {
            //BFS Shortest Path from bot -> cell with highest P(L)
            maxProb = 0.0;
            maxProbCell = null;
            for (Cell cell : openCells) {
                if (maxProb < cell.getProbLeak()) {
                    maxProb = cell.getProbLeak();
                    maxProbCell = cell;
                }
            }
            LinkedList<Cell> shortestPath = Bfs.SP_BFS(bot, maxProbCell);
            if (shortestPath == null) {
                numActions = null;
                return;
            }
            shortestPath.removeFirst();

            //move the bot one step toward cell with highest P(L)
            Cell neighbor = shortestPath.removeFirst();
            bot.isBot = false;
            neighbor.isBot = true;
            bot = neighbor;
            numActions++;
            Bfs.updateDistances(bot);

            //if bot reached one of the leaks and both have not been plugged
            if (bot.isLeak && leak1.isLeak && leak2.isLeak)
                bot.isLeak = false;

            bot.noLeak = true;
            updateProb_Step_Bot8(bot);

            //Sense Action
            if (bot.equals(maxProbCell)) {
                if (performSense) {
                    if (leak1.isLeak && leak2.isLeak)
                        probSenseAction_Bot8(bot, leak1, leak2);
                    else if (leak1.isLeak)
                        probSenseAction(bot, leak1);
                    else
                        probSenseAction(bot, leak2);
                }
                performSense = !performSense;
            }
        }

        Cell leak;
        if (leak1.isLeak) leak = leak1;
        else leak = leak2;

        //run bot 4 algorithm
        performSense = true;
        while (!bot.isLeak) {
            //BFS Shortest Path from bot -> cell with highest P(L)
            maxProb = 0.0;
            maxProbCell = null;
            for (Cell cell : openCells) {
                if (maxProb < cell.getProbLeak()) {
                    maxProb = cell.getProbLeak();
                    maxProbCell = cell;
                }
            }
            LinkedList<Cell> shortestPath = Bfs.SP_BFS(bot, maxProbCell);
            if (shortestPath == null) {
                numActions = null;
                return;
            }
            shortestPath.removeFirst();

            //move the bot one step toward cell with highest P(L)
            Cell neighbor = shortestPath.removeFirst();
            bot.isBot = false;
            neighbor.isBot = true;
            bot = neighbor;
            numActions++;
            Bfs.updateDistances(bot);

            if (bot.isLeak) return;

            bot.noLeak = true;
            updateProb_Step(bot);

            //Sense Action
            if (bot.equals(maxProbCell)) {
                if (performSense) probSenseAction(bot, leak);
                performSense = !performSense;
            }
        }
    }

}