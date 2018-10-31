/**
 * 
 */
package CS2NN16;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * @author shsmchlr
 * This a multi layer network, comprising a hidden layer of neurons with sigmoid activation
 * Followed by another layer with linear/sigmoid activation, or be another multi layer network
 * A layer is defined as a set of neurons which have the same inputs
 */
public class MultiLayerNetwork extends SigmoidLayerNetwork {
	LinearLayerNetwork nextLayer;			// this is the next layer of neurons
	
	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 * @param nextL		the next layer in the network
	 */
	public MultiLayerNetwork(int numIns, int numOuts, DataSet data, LinearLayerNetwork nextL) {
		super(numIns, numOuts, data);			// construct the current layer
		nextLayer = nextL;						// store link to next layer
	}
	/**
	 * calcOutputs of network
	 * @param nInputs	
	 * 
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		// you write code here
	}
	
	/**
	 * outputsToDataSet of the network to the data set
	 * @param ct	
	 * @param d		
	 */
	protected void outputsToDataSet (int ct, DataSet d) {
		// you write code here ... note DataSet wants output(s) of final layer only
	}
	
	/**
	 * find the deltas in the whole network 
	 *	@param errors 	
	 */
	protected void findDeltas(ArrayList<Double> errors) {
		// you write this
	}
	
	/**
	 * change all the weights in the network, in this layer and the next
	 * @param ins		array list of the inputs to the neuron
	 * @param learnRate	learning rate: change is learning rate * input * delta
	 * @param momentum	momentum constant : change is also momentun * change in weight last time
	 */
	protected void changeAllWeights(ArrayList<Double> ins, double learnRate, double momentum) {
		// you write this
	}	
	
	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * @param wtsSplit
	 */
	protected void setWeights (String[] wtsSplit) {
		super.setWeights(wtsSplit);					// copy relevant weights in this layer
		nextLayer.setWeights(Arrays.copyOfRange(wtsSplit, weights.size(), wtsSplit.length));
				// copy remaining strings in wtsSplit and pass to next layer
	}
	/**
	 * Load the weights with random values
	 * @param rgen	random number generator
	 */
	public void setWeights (Random rgen) {
		super.setWeights(rgen);			// do so in this layer
		nextLayer.setWeights(rgen);		// and in next
	}
	/**
	 * return how many weights there are in the network
	 * @return
	 */
	public int numWeights() {
		return 0; 		// change this
	}
	/**
	 * return the weights in the whole network as a string
	 * @return the string
	 */
	public String getWeights() {
		return "";		// change this
	}
	/**
	 * initialise network before running
	 */
	public void doInitialise() {
		super.doInitialise();					// initialise this layer 
		nextLayer.doInitialise();				// and then initialise next layer
	}
	
	/**
	 * function to test MLP on xor problem
	 */
	public static void TestXOR() {
		DataSet Xor = new DataSet("2 1 %.0f %.0f %.3f;x1 x2 XOR;0 0 0;0 1 1;1 0 1;1 1 0");
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Xor, new SigmoidLayerNetwork(2, 1, Xor));
		MLN.setWeights("0.862518 -0.155797 0.282885 0.834986 -0.505997 -0.864449 0.036498 -0.430437 0.481210");
		MLN.doInitialise();
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());
		System.out.println(MLN.doLearn(2000, 0.5,  0.8));
		System.out.println(MLN.doPresent());
		System.out.println("Weights " + MLN.getWeights());
	}
	/**
	 * function to test MLP on other non linear separable problem
	 */
	public static void TestOther() {
		DataSet Other = new DataSet(DataSet.GetFile("other.txt"));
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 2, Other, new SigmoidLayerNetwork(2, 2, Other));
			MLN.presentDataSet(Other);
			MLN.doInitialise();
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
			System.out.println(MLN.doLearn(2000,  0.5,  0.8));
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
		
	}
	/**
	 * function to test MLP on other non linear separable problem using three layers
	 */
	public static void TestThree() {
		DataSet Other = new DataSet(DataSet.GetFile("other.txt"));
		MultiLayerNetwork MLN = new MultiLayerNetwork(2, 4, Other,
										new MultiLayerNetwork (4, 3, Other,
												new SigmoidLayerNetwork(3, 2, Other)) );
			MLN.presentDataSet(Other);
			MLN.doInitialise();
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
			System.out.println(MLN.doLearn(1000,  0.2,  0.6));
			System.out.println(MLN.doPresent());
			System.out.println("Weights " + MLN.getWeights());
		
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		TestXOR();				// test MLP on the XOR problem
	//	TestOther();			// test MLP on the other problem
	//	TestThree();			// test that have 3 hidden layers
	}

}
