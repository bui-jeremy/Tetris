package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.linalg.Shape;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;
import java.lang.reflect.Method;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;
    private double epsilon = 1.0;
    private final double minEpsilon = 0.05;
    private final double epsilonDecay = 0.995;


    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

@Override
public Model initQFunction() {
    final int inputDim = 225; // Match the number of features in getQFunctionInput()
    final int hiddenDim = 32; // Hidden layer size
    final int outDim = 1;     // Output size (Q-value)

    Sequential qFunction = new Sequential();
    qFunction.add(new Dense(inputDim, hiddenDim));
    qFunction.add(new ReLU());
    qFunction.add(new Dense(hiddenDim, outDim));

    return qFunction;
}



    /**
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?
     */
@Override
public Matrix getQFunctionInput(final GameView game, final Mino potentialAction) {
    Board board = game.getBoard();
    Matrix grayscaleImage;
    try {
        grayscaleImage = game.getGrayscaleImage(potentialAction).flatten();
    } catch (Exception e) {
        e.printStackTrace();
        System.exit(-1);
        return null;
    }

    int numElements = grayscaleImage.numel();
    double[] grayscaleData = new double[numElements];
    for (int i = 0; i < numElements; i++) {
        grayscaleData[i] = grayscaleImage.get(0, i);
    }

    int[] columnHeights = calculateColumnHeights(board);
    int aggregateHeight = 0;
    int bumpiness = 0;

    for (int i = 0; i < columnHeights.length; i++) {
        aggregateHeight += columnHeights[i];
        if (i > 0) bumpiness += Math.abs(columnHeights[i] - columnHeights[i - 1]);
    }

    int numHoles = calculateNumberOfHoles(board, columnHeights);

    // Compute cleared lines
    int clearedLines = 0;
    try {
        Method method = Board.class.getDeclaredMethod("getFullLines");
        method.setAccessible(true);
        clearedLines = ((List<?>) method.invoke(board)).size();
    } catch (Exception e) {
        e.printStackTrace();
    }

    // Compute weighted aggregate height
    int weightedAggregateHeight = 0;
    for (int i = 0; i < columnHeights.length; i++) {
        weightedAggregateHeight += columnHeights[i] * (i + 1);
    }

    // Combine features
    double[] features = new double[grayscaleData.length + 5];
    System.arraycopy(grayscaleData, 0, features, 0, grayscaleData.length);
    features[grayscaleData.length] = (double) aggregateHeight / (Board.NUM_ROWS * Board.NUM_COLS);
    features[grayscaleData.length + 1] = (double) bumpiness / Board.NUM_COLS;
    features[grayscaleData.length + 2] = (double) numHoles / (Board.NUM_ROWS * Board.NUM_COLS);
    features[grayscaleData.length + 3] = (double) clearedLines / Board.NUM_ROWS;
    features[grayscaleData.length + 4] = (double) weightedAggregateHeight / (Board.NUM_ROWS * Board.NUM_COLS);

    Matrix featureMatrix = Matrix.full(1, features.length, 0.0);
    for (int i = 0; i < features.length; i++) {
        featureMatrix.set(0, i, features[i]);
    }

    return featureMatrix;
}



    private int[] calculateColumnHeights(Board board) {
        int[] heights = new int[Board.NUM_COLS];
        for (int x = 0; x < Board.NUM_COLS; x++) {
            for (int y = Board.NUM_ROWS - 1; y >= 0; y--) {
                if (board.isCoordinateOccupied(x, y)) {
                    heights[x] = Board.NUM_ROWS - y;
                    break;
                }
            }
        }
        return heights;
    }


    private int calculateNumberOfHoles(Board board, int[] columnHeights) {
        int holes = 0;
        for (int x = 0; x < Board.NUM_COLS; x++) {
            for (int y = Board.NUM_ROWS - columnHeights[x]; y < Board.NUM_ROWS; y++) {
                if (!board.isCoordinateOccupied(x, y)) {
                    holes++;
                }
            }
        }
        return holes;
    }


    /**
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game, final GameCounter gameCounter) {
        // Example: Adjust exploration every 50 games
        if (gameCounter.getCurrentGameIdx() % 50 == 0) {
            // You can use gameCounter.getCurrentGameIdx() to conditionally adjust epsilon
            epsilon = Math.max(minEpsilon, epsilon * epsilonDecay);
            System.out.println("Updated epsilon: " + epsilon);
        }
        return this.getRandom().nextDouble() < epsilon;
    }


    /**
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game) {
        List<Mino> possibleMoves = game.getFinalMinoPositions();
        int randIdx = this.getRandom().nextInt(possibleMoves.size());
        return possibleMoves.get(randIdx);
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset, LossFunction lossFunction, Optimizer optimizer, long numUpdates) {
        for (int epochIdx = 0; epochIdx < numUpdates; ++epochIdx) {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix>> batchIterator = dataset.iterator();

            while (batchIterator.hasNext()) {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(), lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch (Exception e) {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
public double getReward(final GameView game) {
    double reward = game.getScoreThisTurn();
    Board board = game.getBoard();

    int[] columnHeights = calculateColumnHeights(board);
    int maxHeight = 0;
    int bumpiness = 0;
    int deadZones = 0;

    for (int i = 0; i < columnHeights.length; i++) {
        maxHeight = Math.max(maxHeight, columnHeights[i]);
        if (i > 0) bumpiness += Math.abs(columnHeights[i] - columnHeights[i - 1]);
    }

    int numHoles = calculateNumberOfHoles(board, columnHeights);
    for (int x = 0; x < Board.NUM_COLS; x++) {
        for (int y = 0; y < columnHeights[x]; y++) {
            if (!board.isCoordinateOccupied(x, y)) {
                deadZones++;
            }
        }
    }

    int clearedLines = 0;
    try {
        Method method = Board.class.getDeclaredMethod("getFullLines");
        method.setAccessible(true);
        clearedLines = ((List<?>) method.invoke(board)).size();
    } catch (Exception e) {
        e.printStackTrace();
    }

    // Reward/Penalty based on board state
    reward += clearedLines * (clearedLines == 4 ? 800 : 500); // Reward for Tetris
    reward -= maxHeight * 2;      // Penalize height
    reward -= numHoles * 50;      // Penalize holes
    reward -= bumpiness * 10;     // Penalize uneven terrain
    reward -= deadZones * 50;     // Penalize dead zones

    return reward;
}




}
