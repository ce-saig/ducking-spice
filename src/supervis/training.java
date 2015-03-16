package supervis;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class training {

	public static void main(String args[]) {

		try {
			FileReader trainreader = new FileReader("crop_price.arff");
			FileReader testreader = new FileReader("crop_price.arff");

			Instances train = new Instances(trainreader);
			Instances test = new Instances(testreader);
			train.setClassIndex(train.numAttributes() - 1);
			test.setClassIndex(test.numAttributes() - 1);

			MultilayerPerceptron mlp = new MultilayerPerceptron();
			mlp.setOptions(Utils
					.splitOptions("-L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 4"));

			mlp.buildClassifier(train);

			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(mlp, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n",
					false));
			trainreader.close();
			testreader.close();
			
			
			System.out.println("-------------------------");
			double[] m=mlp.distributionForInstance(train.instance(1));
			for (int i = 0; i < m.length; i++) {
				System.out.println(m[i]);
			}
			
			/*
			System.out.println("-------------------------");
			System.out.println(train.instance(1).attribute(3).value((int) mlp.classifyInstance(train.instance(1))));
			System.out.println(train.instance(1).attribute(3).value((int) mlp.classifyInstance(train.instance(2))));
			System.out.println(train.instance(1).attribute(3).value((int) mlp.classifyInstance(train.instance(259))));
			*/
			
			System.out.println("-------------------------");
			
				
			//New instances
			Instance example = new Instance(4);
//			example.setValue(train.attribute(0), train.attribute(0).indexOfValue("1"));
//			example.setValue(train.attribute(1), train.attribute(1).indexOfValue("1"));
//			example.setValue(train.attribute(2), train.attribute(1).indexOfValue("1"));
			
			example.setValue(0, 0);
			example.setValue(1, 0);
			example.setValue(2, 0);
			//example.setValue(attr3, --);
			
			example.setDataset(train);
			
			System.out.println(example);
			
			System.out.println(train.instance(1).attribute(3).value((int) mlp.classifyInstance(example)));
			System.out.println( mlp.classifyInstance(example));


		} catch (Exception ex) {

			ex.printStackTrace();

		}
	}

}