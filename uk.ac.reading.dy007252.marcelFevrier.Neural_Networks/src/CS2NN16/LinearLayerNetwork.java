/**
 * 
 */
package CS2NN16;

import java.util.ArrayList;
import java.util.Random;

/**
 * @author shsmchlr
 * This is a class for a layer of neurons with linear activation
 * All such neurons share the same inputs.
 */
public class LinearLayerNetwork {
	/**
	 * data are arraylists of weights and the change in weights
	 * and of the outputs and deltas 
	 * and also how many inputs, neurons and weights
	 * also has data set used with network
	 */
	protected ArrayList<Double> weights;
	protected ArrayList<Double> changeInWeights;
	protected ArrayList<Double> outputs;
	protected ArrayList<Double> deltas;
	protected int numInputs, numNeurons, numWeights;
	protected DataSet trainData;
	
	/**
	 * Constructor for neuron
	 * @param numIns	how many inputs there are (hence how many weights needed)
	 * @param numOuts	how many outputs there are (hence how many neurons needed)
	 * @param data		the data set used to train the network
	 */
	public LinearLayerNetwork(int numIns, int numOuts, DataSet data) {
		numInputs = numIns;							// store number inputs
		numNeurons = numOuts;						// and of outputs in object
		numWeights = (numInputs + 1) * numNeurons;	// for convenience calculate number of weights
				// each neuron has numInputs + 1 weights (+1 because of bias weight)

		weights = new ArrayList<Double>();			// create array list for weights
		changeInWeights = new ArrayList<Double>();	// and for the change in weights
		outputs = new ArrayList<Double>();			// create array list for outputs
		deltas = new ArrayList<Double>();			// and for the change in deltas
		for (int ct=0; ct<numWeights; ct++) {		// for each weight
			weights.add(0.0);						// add next weight as 0
			changeInWeights.add(0.0);				// add next change in weight, value 0
		}
		for (int ct=0; ct<numNeurons; ct++) {		// for each neuron
			outputs.add(0.0);						// add a zero output
			deltas.add(0.0);						// add a zero delta
		}
		trainData = data;							// remember data set used for training
	}
	/**
	 * calcOutputs of neuron
	 * @param nInputs	arraylist with the neuron inputs
	 * Calculates weighted sum being weight(0) + inputs(0..n) * weights(1..n+1)
	 */
	protected void calcOutputs(ArrayList<Double> nInputs) {
		int wtIndex = 0;									// used to index weights in order
		double output;
		for (int neuronct = 0; neuronct<numNeurons; neuronct++) {
			output = weights.get(wtIndex++);						// start with bias weight( * 1)
			for (int inputct=0; inputct<numInputs; inputct++) 		// for remaining weights
				 output += nInputs.get(inputct) * weights.get(wtIndex++);
											// add weight*appropriate input and move to next weight
			outputs.set(neuronct, output);			// set calculated output as the neuron output
		}				
	}
	/**
	 * outputsToDataSet to the given data set
	 * @param ct	which item in the data set
	 * @param d		the data set
	 */
	protected void outputsToDataSet (int ct, DataSet d) {
		d.storeOutputs(ct, outputs);							// just store outputs in data set
	}
	
	/**
	 * compute outputs of network by passing it each item in data set in turn, 
	 * these outputs are put back into the data set
	 * @param d	data set
	 */
	public void presentDataSet(DataSet d) {
		for (int ct=0; ct < d.numInSet(); ct++) {			// for each item in data set
			calcOutputs(d.getIns(ct));						// calculate output
			outputsToDataSet(ct, d);							// and put in data set
		}
	}
	
	/**
	 * find deltas 
	 *	@param errors	
	 */
	protected void findDeltas(ArrayList<Double> errors) {
		for (int cnt = 0; cnt < errors.size();cnt++){ // for each error in errors
			deltas.set(cnt, errors.get(cnt)); // set the delta (of the same index as the error's index) as the error
		}
	}	
	
	/**
	 * return index into weights array list of the wWeight'th weight of the wNeuron'th neuron
	 * @param wNeuron the index of the neuron
	 * @param wWeight the index of the weight
	 * @return the actual index in weights for this particular weight in this particular neuron
	 */
	private int weightIndex (int wNeuron, int wWeight) {
		/*
		 * the index of a weight for a particular neuron is that neuron's index 
		 * multiplied by the number of weights per neuron 
		 * plus the weight itself 
		 */
		
		return wNeuron * (numWeights / numNeurons) + wWeight; 
	}
	
	/**
	 * change a given weight 
	 * @param wNeuron the index of the weight's neuron
	 * @param wWeight the weight's index
	 * @param theIn	the input for the weight		
	 * @param learnRate the learning rate
	 * @param momentum	the amount of momentum
	 */
	private void changeOneWeight(int wNeuron, int wWeight, double theIn, double learnRate, double momentum) {

		/*
		 * the new wWeight is the index of that particular weight for its respective neuron
		 */
		wWeight = weightIndex(wNeuron, wWeight);
		
		/*
		 * the change in weight (wWeight) is the learning rate 
		 * multiplied by the delta of that particular neuron 
		 * multiplied by the input of that particular neuron
		 * plus the momentum multiplied by the previous change in weight
		 */		
		changeInWeights.set(wWeight, (learnRate * deltas.get(wNeuron) * theIn + momentum*changeInWeights.get(wWeight)));
		
		/*
		 * After calculating the new change in weight the current weight 
		 * is equal to itself plus the new change in weight
		 */	
		weights.set(wWeight, weights.get(wWeight) + changeInWeights.get(wWeight));
		
	}
	
	/**
	 * change all the weights in layer
	 * @param ins the list of inputs
	 * @param learnRate	the learning rate
	 * @param momentum	the amount of momentum
	 */
	protected void changeAllWeights(ArrayList<Double> ins, double learnRate, double momentum) {
		for(int neuron = 0; neuron < numNeurons; neuron++){ // for every neuron
			changeOneWeight(neuron, 0, 1, learnRate, momentum); // calculate the change in weight for the bias weight of each neuron
			for (int weight = 1; weight <= ins.size(); weight++){ // for each weight between each input and the neuron (excluding the bias weight because that has already been accounted for)
				changeOneWeight(neuron, weight, ins.get(weight-1), learnRate, momentum); // change a particular weight of the neuron using the input that the weight is connected to
			}
		}
	}
	
	/**
	 * adapt the network, by inputting each item from the data set in turn, calculating
	 * the output, the error and delta, and adjusting all the weights
	 * @param d			data set
	 * @param learnRate	learning rate constant
	 * @param momentum	momentum constant
	 */
	public void learnDataSet(DataSet d, double learnRate, double momentum) {
		for (int ct=0; ct < d.numInSet(); ct++) {				// for each item in set
			calcOutputs(d.getIns(ct));							// calc outputs
			outputsToDataSet(ct, d);								// put in data set
			findDeltas(d.getErrors(ct));						// calc deltas, from the errors
			changeAllWeights(d.getIns(ct), learnRate, momentum);// change the weights
		}
		d.addToSSELog();
	}
	
	/**
	 * return the array list containing the outputs of this layer of neurons
	 * @return
	 */
	protected ArrayList<Double> getOutputs() {
		return outputs;
	}
	
	/**
	 * Calculate the errors in the previous layer, being the deltas in this layer * associated weights
	 * this is used in the back propagation algorithm
	 * 
	 * @return	arraylist of errors
	 */
	
	/*
	public ArrayList<Double> weightedDeltas() {
		ArrayList<Double> wtDeltas = new ArrayList<Double>();	// create array for answer
		for (int ct = 0; ct < numInputs; ct++) { // for all neurons in previous layer (num of inputs in this layer)
			wtDeltas.add(this.deltas.get(0) * this.weights.get(ct + 1)); // multiply delta with respective weight
		}
		return wtDeltas;
	}
	*/
	
	/**
	 * Calculate the errors in the previous layer, being the deltas in this layer * associated weights
	 * this is used in the back propagation algorithm
	 * @return errors the array of errors for the previous layer
	 */
	public ArrayList<Double> weightedDeltas(){
		ArrayList<Double> errors = new ArrayList<Double>(); 				// the array for the previous neuron's errors
		double error; 														// will store the error of each neuron
		int index; 															// the index of the respective weight 
		for (int prevNeuron = 0; prevNeuron < numInputs; prevNeuron++) { 	// for each neuron in the previous layer
			error = 0; 														// initialise error to 0
			for (int thisNeuron = 0; thisNeuron < this.numNeurons; thisNeuron++) { // for each neuron in this layer
				index = this.weightIndex(thisNeuron, prevNeuron + 1); 		// the index of the weight from the previous layer to the neuron in this layer
				error += this.deltas.get(thisNeuron) * this.weights.get(index); // calc and add this error to the total error for the previous neuron
			}
			errors.add(error); 												// add the error to the list of errors
		}
		return errors;
	}
	
	/**
	 * Load weights with the values in the array of strings wtsSplit
	 * @param wtsSplit
	 */
	protected void setWeights (String[] wtsSplit) {
		for (int ct=0; ct<weights.size(); ct++) weights.set(ct, Double.parseDouble(wtsSplit[ct])); 
	}			// for each item, set weight by converting string to double
	/**
	 * Load the weights with the values in the String wts
	 * @param wts
	 */
	public void setWeights (String wts) {
		setWeights(wts.split(" "));			// split string into array of string and so set weights
	}
	/**
	 * Load the weights with random values in range -1 to 1
	 * @param rgen	random number generator
	 */
	public void setWeights (Random rgen) {
		for (int ct=0; ct<weights.size(); ct++) weights.set(ct,2.0*rgen.nextDouble() - 1);
	}
	/**
	 * return how many weights there are in the neuron
	 * @return
	 */
	public int numWeights() {
		return weights.size();
	}
	/**
	 * return all the weights in the layer as a string each separated by spaces
	 * @return s the string representation of all weights
	 */
	public String getWeights() {
		String s = "";
		for (Double w:weights){ // for each weight in weights
			s += String.format("%.5f", w) + " "; // Append to s the value of weight w which is formatted to 5 decimal points
		}
		return s;										// return the result
	}
	/**
	 * initialise network before running
	 */
	public void doInitialise() {
		for (int ct=0; ct<changeInWeights.size(); ct++) changeInWeights.set(ct, 0.0);
				// set the change in weights to be 0
		trainData.clearSSELog();
	}
	/**
	 * present the data to the network and return string describing result
	 * @return
	 */
	public String doPresent() {
		presentDataSet(trainData);
		return trainData.toString(true, true) + "\nOver Set : " + trainData.dataAnalysis()+"\n";
	}

	
	/**
	 * create string which says Epoch then adds the actual epoch in a fixed width field
	 * @param epoch
	 * @return
	 */
	protected String addEpochString (int epoch) {
		return "Epoch " + String.format("%4d", epoch);
	}
	
	/**
	 * get network to learn for numEpochs
	 * @param numEpochs		number of epochs to learn
	 * @param lRate			learning rate
	 * @param momentum		momentum
	 * @return				String with data about learning eg SSEs at relevant epochs
	 * 						At each epoch if numEpochs low, or do so at 10 of the epochs
	 */
	public String doLearn (int numEpochs, double lRate, double momentum) {
		int epochsSoFar = trainData.sizeSSELog();		// SSE log indicates how many epochs so far
		String s = "";
		for (int ct=1; ct<=numEpochs; ct++) {			// for n epochs
			learnDataSet(trainData, lRate, momentum);	// present data and adapt weights
			if (numEpochs<20 || ct % (numEpochs/10) == 0) // print appropriate number of times
				s = s + addEpochString(ct+epochsSoFar) + " : " + trainData.dataAnalysis()+"\n";
		}				// Epoch, and SSE, and if appropriate % correctly classified
		return s;
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// Test network on example data set
		DataSet AndOrXor = new DataSet("2 3 %.0f %.0f %.3f;x1 x2 AND OR XOR;0 0 0 0 0;0 1 0 1 1;1 0 0 1 1;1 1 1 1 0");
		LinearLayerNetwork LN = new LinearLayerNetwork(2, 3, AndOrXor);
		LN.setWeights("0.2 0.5 0.3 0.3 0.5 0.1 0.4 0.1 0.2");
		LN.doInitialise();
		System.out.println(LN.doPresent());
		System.out.println("Weights " + LN.getWeights());
		System.out.println(LN.doLearn(7,  0.2,  0.1));
		System.out.println(LN.doPresent());
		System.out.println("Weights " + LN.getWeights());
		
	}

}
