package cs276.pa4;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

public class PairwiseLearner extends Learner {
  private LibSVM model;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
	@Override
	public Instances extract_train_features(String train_data_file,
			String train_rel_file, Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

	@Override
	public Classifier training(Instances dataset) {
		try {
			this.model.buildClassifier(dataset);
		} catch (Exception e) {
			e.printStackTrace();
		}
		//System.out.println("Num of Param: " + this.model.numParameters());
        //System.out.println("Weights:");
        //for (double coefficient : this.model.coefficients()) {
		//	System.out.println(coefficient);
		//}
		//System.out.println("Slope: " + this.model.getSlope());
        //System.out.println("Intercept: " + this.model.getIntercept());
       // System.out.println(this.model);
		return this.model;
	}

	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) {
		/*
		 * @TODO: Your code here
		 */
		return null;
	}

	@Override
	public Map<String, List<String>> testing(TestFeatures tf,
			Classifier model) {
		/*
		 * @TODO: Your code here
		 */
		return null;
	}
	
	//TODO: Remove. Only for test
	public static void testRegressionModel() throws Exception{
	
		Instances data = new Instances(new BufferedReader(new FileReader("libsvm.arff")));
		data.setClassIndex(data.numAttributes() - 1);
		//build model
		LibSVM model = new LibSVM();
		//NaiveBayes model = new NaiveBayes();
		model.buildClassifier(data); //the last instance with missing class is not used
		System.out.println(model);
		System.out.println("Num of Param: " + model);
        System.out.println("Weights:");
        for (double coefficient : model.coefficients()) {
			System.out.println(coefficient);
		}
		//classify the last instance
		Instance myHouse = data.lastInstance();
		double price = model.classifyInstance(myHouse);
		System.out.println("My house ("+myHouse+"): "+price);
		
	}
	
	//TODO: Remove. Only for test
	public static void main(String[] args) throws Exception{
//		for(int i=0; i<10; i++) {
//			System.out.println((int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100)+","+(int)(Math.random()*100000+1000000));
//		}
		//PointwiseLearner.testRegressionModel();
		PairwiseLearner.testRegressionModel();
	}

}
