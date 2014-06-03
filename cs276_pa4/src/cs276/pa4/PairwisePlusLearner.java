package cs276.pa4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import cs276.pa4.Document;
import cs276.pa4.Query;
import edu.stanford.cs276.util.Pair;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

<<<<<<< HEAD
public class PairwisePlusLearner extends Learner {
  private LibSVM model;
  public PairwisePlusLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwisePlusLearner(double C, double gamma, boolean isLinearKernel){
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
			String train_rel_file, Map<String, Double> idfs) throws Exception{
		
		TestFeatures tf = new TestFeatures("plusPoint");
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("bm25"));
		attributes.add(new Attribute("pagerank"));
		attributes.add(new Attribute("smallestwindow"));
		attributes.add(new Attribute("relevance_score"));
		dataset = new Instances("train_dataset", attributes, 0);
		
		/* Add data */
		Map<Query,List<Document>> queryDocs = Util.loadTrainData(train_data_file);
		Map<String, Map<String, Double>> queryDocScore = Util.loadRelData(train_rel_file);
		Map<Query,Map<String, Document>> queryDocsBm25 = loadTrainData(train_data_file);
		BM25Scorer bm25Scorer = new BM25Scorer(queryDocsBm25);
		SmallestWindowScorer windowScorer = new SmallestWindowScorer(idfs, queryDocsBm25);
		
		for(Query q: queryDocs.keySet()){
			String query = q.query;
			//System.out.println(query);
			for(Document d: queryDocs.get(q)){
				Map<String,Map<String, Double>> tfDoc = AScorer.getDocTermFreqs(d, q);
				
				
				double tfIdfUrl = 0.0;
				double tfIdfTitle = 0.0;
				double tfIdfBody = 0.0;
				double tfIdfHeader = 0.0;
				double tfIdfAnchor = 0.0;
				double relevanceScore = 0.0;
				double bm25 = bm25Scorer.getSimScore(d, q, idfs);
				double pagerank = d.page_rank;
				double smallwindow = windowScorer.getSimScore(d, q, idfs);
				
				for(String term: q.words){
					double qIDF = Util.IDF(term, idfs);
					double d_tf_url = getDocFieldTF(term, "url", tfDoc);
					tfIdfUrl = tfIdfUrl + qIDF * d_tf_url;
=======
public class PairwisePlusLearner extends Learner{

	  private LibSVM model;
	  boolean useBM25 = false;
	  boolean useSW = false;
	  boolean usePR = false;
	  BM25Scorer bm25;
	  SmallestWindowScorer sw;
	  
	  public PairwisePlusLearner(boolean isLinearKernel){
	    try{
	      model = new LibSVM();
	    } catch (Exception e){
	      e.printStackTrace();
	    }
	    
	    if(isLinearKernel){
	      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
	    }
	  }
	  
	  public PairwisePlusLearner(boolean isLinearKernel, boolean useBM25, boolean useSW, boolean usePR, String train_data_file) throws Exception{
		  this.useBM25 = useBM25;
		  this.usePR = usePR;
		  this.useSW = useSW;
		  Map<Query,Map<String, Document>> queryDict = Util.loadQueryDict(train_data_file);
		    try{
		      model = new LibSVM();
		    } catch (Exception e){
		      e.printStackTrace();
		    }
		    
		    if(isLinearKernel){
		      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		    }
		    
		    if(useBM25){
		    	bm25 = new BM25Scorer(queryDict);
		    }
	  }
	  
	  public PairwisePlusLearner(double C, double gamma, boolean isLinearKernel){
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
				String train_rel_file, Map<String, Double> idfs) throws Exception{
			
			
			double bm25Score = 0.0;
			double sqScore = 0.0;
			double pageRank = 0.0;
			
			TestFeatures tf = new TestFeatures("point");
			Instances dataset = null;
			
			/* Build attributes list */
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			attributes.add(new Attribute("url_w"));
			attributes.add(new Attribute("title_w"));
			attributes.add(new Attribute("body_w"));
			attributes.add(new Attribute("header_w"));
			attributes.add(new Attribute("anchor_w"));
			attributes.add(new Attribute("relevance_score"));
			dataset = new Instances("train_dataset", attributes, 0);
			
			/* Add data */
			Map<Query,List<Document>> queryDocs = Util.loadTrainData(train_data_file);
			Map<String, Map<String, Double>> queryDocScore = Util.loadRelData(train_rel_file);
			
			for(Query q: queryDocs.keySet()){
				String query = q.query;
				//System.out.println("Query: "+query);
				for(Document d: queryDocs.get(q)){
>>>>>>> FETCH_HEAD
					
					//System.out.println("\tDoc: "+d.url);
					Map<String,Map<String, Double>> tfDoc = AScorer.getDocTermFreqs(d, q);
					
					//Testing BM25
					 if(useBM25){
						 bm25Score = bm25.getSimScore(d, q, idfs);
						 //System.out.println("\tBM25 Score = "+bm25Score+"\n");
					 }
					
					
					double tfIdfUrl = 0.0;
					double tfIdfTitle = 0.0;
					double tfIdfBody = 0.0;
					double tfIdfHeader = 0.0;
					double tfIdfAnchor = 0.0;
					double relevanceScore = 0.0;
					
					for(String term: q.words){
						double qIDF = Util.IDF(term, idfs);
						double d_tf_url = getDocFieldTF(term, "url", tfDoc);
						tfIdfUrl = tfIdfUrl + qIDF * d_tf_url;
						
						double d_tf_title = getDocFieldTF(term, "title", tfDoc);
						tfIdfTitle = tfIdfTitle + qIDF * d_tf_title;
						
						double d_tf_body = getDocFieldTF(term, "body", tfDoc);
						tfIdfBody = tfIdfBody + qIDF * d_tf_body;
						
						double d_tf_header = getDocFieldTF(term, "header", tfDoc);
						tfIdfHeader = tfIdfHeader + qIDF * d_tf_header;
						
						double d_tf_anchor = getDocFieldTF(term, "anchor", tfDoc);
						tfIdfAnchor = tfIdfAnchor + qIDF * d_tf_anchor;
					}
					
					//Getting Relevance Score
					String url = d.url;
					Map<String, Double> urlRelScore = queryDocScore.get(q.query);
					if(urlRelScore.containsKey(d.url)){
						relevanceScore = urlRelScore.get(d.url);
					//	System.out.println("\t"+url+": "+relevanceScore);
					}
					double[] instance = new double[6];
					instance[0] = tfIdfUrl;
					instance[1] = tfIdfTitle;
					instance[2] = tfIdfBody;
					instance[3] = tfIdfHeader;
					instance[4] = tfIdfAnchor;
					instance[5] = relevanceScore;
					dataset.add(new DenseInstance(1.0, instance));
					tf.add(query, url, new DenseInstance(1.0, instance));
				}
			}
			
			//System.out.println(tf.features);
			/* Set last attribute as target */
			dataset.setClassIndex(dataset.numAttributes() - 1);
			Standardize filter = new Standardize();
			filter.setInputFormat(dataset);
			Instances new_dataset = Filter.useFilter(dataset, filter);
			tf.StandardizeFeatures();
			//System.out.println(tf.features);
			
			
			Instances datasetPair = null;
			
			/* Build attributes list */
			ArrayList<Attribute> attributesPair = new ArrayList<Attribute>();
			attributesPair.add(new Attribute("url_w"));
			attributesPair.add(new Attribute("title_w"));
			attributesPair.add(new Attribute("body_w"));
			attributesPair.add(new Attribute("header_w"));
			attributesPair.add(new Attribute("anchor_w"));
			attributesPair.add(new Attribute("class",Arrays.asList("-1","+1")));
			datasetPair = new Instances("train_dataset_pair", attributesPair, 0);
			//System.out.println("start merge:");
			
			for(Query q: queryDocs.keySet()){
				String query = q.query;
				//for(Document d: queryDocs.get(q)){
				for(int i=0; i<queryDocs.get(q).size()-1;i++) {
					for (int j = i+1; j<queryDocs.get(q).size(); j++) {
						String prevUrl = queryDocs.get(q).get(i).url;
						String currUrl = queryDocs.get(q).get(j).url;
						Instance prev = tf.getInstance(query, prevUrl);
						Instance curr = tf.getInstance(query, currUrl);
						if(prev.value(5)!=curr.value(5)) {
							String C = prev.value(5)>curr.value(5) ? "+1" : "-1";
							double url = prev.value(0)-curr.value(0);
							double title = prev.value(1)-curr.value(1);
							double body = prev.value(2)-curr.value(2);
							double header = prev.value(3)-curr.value(3);
							double anchor = prev.value(4)-curr.value(4);
							Instance merge = new DenseInstance(6);
							merge.setDataset(datasetPair);
							merge.setValue(0, url);
							merge.setValue(1, title);
							merge.setValue(2, body);
							merge.setValue(3, header);
							merge.setValue(4, anchor);
							merge.setValue(5, C);
							datasetPair.add(merge);
							/*
							//For Reverse feature Diff
							String CR = prev.value(5) > curr.value(5) ? "-1" : "+1";
							double urlR = curr.value(0)-prev.value(0);
							double titleR = curr.value(1)-prev.value(1);
							double bodyR = curr.value(2)-prev.value(2);
							double headerR = curr.value(3)-prev.value(3);
							double anchorR = curr.value(4)-prev.value(4);
							Instance mergeR = new DenseInstance(6);
							mergeR.setDataset(datasetPair);
							mergeR.setValue(0, urlR);
							mergeR.setValue(1, titleR);
							mergeR.setValue(2, bodyR);
							mergeR.setValue(3, headerR);
							mergeR.setValue(4, anchorR);
							mergeR.setValue(5, CR);
							datasetPair.add(mergeR);
							*/
						}
					}
				}
<<<<<<< HEAD
				double[] instance = new double[9];
				instance[0] = tfIdfUrl;
				instance[1] = tfIdfTitle;
				instance[2] = tfIdfBody;
				instance[3] = tfIdfHeader;
				instance[4] = tfIdfAnchor;
				instance[5] = bm25;
				instance[6] = pagerank;
				instance[7] = smallwindow;
				instance[8] = relevanceScore;
				dataset.add(new DenseInstance(1.0, instance));
				tf.add(query, url, new DenseInstance(1.0, instance));
=======
				
>>>>>>> FETCH_HEAD
			}
			datasetPair.setClassIndex(datasetPair.numAttributes() - 1);
			//System.out.println(datasetPair);
			return datasetPair;
		}
		
<<<<<<< HEAD
		//System.out.println(tf.features);
		/* Set last attribute as target */
		dataset.setClassIndex(dataset.numAttributes() - 1);
		Standardize filter = new Standardize();
		filter.setInputFormat(dataset);
		Instances new_dataset = Filter.useFilter(dataset, filter);
		tf.StandardizeFeatures();
		//System.out.println(tf.features);
		
		
		Instances datasetPair = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributesPair = new ArrayList<Attribute>();
		attributesPair.add(new Attribute("url_w"));
		attributesPair.add(new Attribute("title_w"));
		attributesPair.add(new Attribute("body_w"));
		attributesPair.add(new Attribute("header_w"));
		attributesPair.add(new Attribute("anchor_w"));
		attributesPair.add(new Attribute("bm25"));
		attributesPair.add(new Attribute("pagerank"));
		attributesPair.add(new Attribute("smallestwindow"));
		attributesPair.add(new Attribute("class",Arrays.asList("-1","+1")));
		datasetPair = new Instances("train_dataset_pair", attributesPair, 0);
		//System.out.println("start merge:");
		
		for(Query q: queryDocs.keySet()){
			String query = q.query;
			//for(Document d: queryDocs.get(q)){
			for(int i=0; i<queryDocs.get(q).size()-1;i++) {
				for (int j = i+1; j<queryDocs.get(q).size(); j++) {
					String prevUrl = queryDocs.get(q).get(i).url;
					String currUrl = queryDocs.get(q).get(j).url;
					Instance prev = tf.getInstance(query, prevUrl);
					Instance curr = tf.getInstance(query, currUrl);
					if(prev.value(8)!=curr.value(8)) {
						String C = prev.value(8)>curr.value(8) ? "+1" : "-1";
						double url = prev.value(0)-curr.value(0);
						double title = prev.value(1)-curr.value(1);
						double body = prev.value(2)-curr.value(2);
						double header = prev.value(3)-curr.value(3);
						double anchor = prev.value(4)-curr.value(4);
						double bm25 = prev.value(5)-curr.value(5);
						double pageRank = prev.value(6)-curr.value(6);
						double window = prev.value(7)-curr.value(7);
						Instance merge = new DenseInstance(9);
						merge.setDataset(datasetPair);
						merge.setValue(0, url);
						merge.setValue(1, title);
						merge.setValue(2, body);
						merge.setValue(3, header);
						merge.setValue(4, anchor);
						merge.setValue(5, bm25);
						merge.setValue(6, pageRank);
						merge.setValue(7, window);
						merge.setValue(8, C);
						datasetPair.add(merge);
						/*
						//For Reverse feature Diff
						String CR = prev.value(5) > curr.value(5) ? "-1" : "+1";
						double urlR = curr.value(0)-prev.value(0);
						double titleR = curr.value(1)-prev.value(1);
						double bodyR = curr.value(2)-prev.value(2);
						double headerR = curr.value(3)-prev.value(3);
						double anchorR = curr.value(4)-prev.value(4);
						Instance mergeR = new DenseInstance(6);
						mergeR.setDataset(datasetPair);
						mergeR.setValue(0, urlR);
						mergeR.setValue(1, titleR);
						mergeR.setValue(2, bodyR);
						mergeR.setValue(3, headerR);
						mergeR.setValue(4, anchorR);
						mergeR.setValue(5, CR);
						datasetPair.add(mergeR);
						*/
					}
				}
=======
		private static double tfIDF(Query q, String type, Map<String,Map<String, Double>> tfDoc, Map<String, Double> idfs){
			double tfId = 0.0;
			for(String term: q.words){
				double qIDF = Util.IDF(term, idfs);
				double d_tf = getDocFieldTF(term, "url", tfDoc);
				tfId = tfId + qIDF * d_tf;
>>>>>>> FETCH_HEAD
			}
			
			
			return tfId;
		}
		
		private static double relevanceScore(Map<String, Double> urlRelScore, Document d){
			double relevanceScore = 0.0;
			if(urlRelScore.containsKey(d.url)){
				relevanceScore = urlRelScore.get(d.url);
			}
			return relevanceScore;
		}
<<<<<<< HEAD
		return relevanceScore;
	}
	
	public static Map<Query,Map<String, Document>> loadTrainData(String feature_file_name) throws Exception {
		File feature_file = new File(feature_file_name);
		if (!feature_file.exists() ) {
			System.err.println("Invalid feature file name: " + feature_file_name);
			return null;
		}
		
		BufferedReader reader = new BufferedReader(new FileReader(feature_file));
		String line = null, url= null, anchor_text = null;
		Query query = null;
		
		/* feature dictionary: Query -> (url -> Document)  */
		Map<Query,Map<String, Document>> queryDict =  new HashMap<Query,Map<String, Document>>();
		
		while ((line = reader.readLine()) != null && !line.isEmpty()) 
		{
			//System.out.println("[DEBUG] line = "+line);
			String[] tokens = line.split(":", 2);
			String key = tokens[0].trim();
			String value = tokens[1].trim();

			if (key.equals("query"))
			{
				query = new Query(value);
				queryDict.put(query, new HashMap<String, Document>());
			} 
			else if (key.equals("url")) 
			{
				url = value;
				queryDict.get(query).put(url, new Document(url));
			} 
			else if (key.equals("title")) 
			{
				queryDict.get(query).get(url).title = new String(value);
			}
			else if (key.equals("header"))
			{
				if (queryDict.get(query).get(url).headers == null)
					queryDict.get(query).get(url).headers =  new ArrayList<String>();
				queryDict.get(query).get(url).headers.add(value);
			}
			else if (key.equals("body_hits")) 
			{
				if (queryDict.get(query).get(url).body_hits == null)
					queryDict.get(query).get(url).body_hits = new HashMap<String, List<Integer>>();
				String[] temp = value.split(" ", 2);
				String term = temp[0].trim();
				List<Integer> positions_int;
				
				if (!queryDict.get(query).get(url).body_hits.containsKey(term))
				{
					positions_int = new ArrayList<Integer>();
					queryDict.get(query).get(url).body_hits.put(term, positions_int);
				} else
					positions_int = queryDict.get(query).get(url).body_hits.get(term);
				
				String[] positions = temp[1].trim().split(" ");
				for (String position : positions)
					positions_int.add(Integer.parseInt(position));
				
			} 
			else if (key.equals("body_length"))
				queryDict.get(query).get(url).body_length = Integer.parseInt(value);
			else if (key.equals("pagerank"))
				queryDict.get(query).get(url).page_rank = Integer.parseInt(value);
			else if (key.equals("anchor_text"))
			{
				anchor_text = value;
				if (queryDict.get(query).get(url).anchors == null)
					queryDict.get(query).get(url).anchors = new HashMap<String, Integer>();
			}
			else if (key.equals("stanford_anchor_count"))
				queryDict.get(query).get(url).anchors.put(anchor_text, Integer.parseInt(value));      
		}

		reader.close();
		
		return queryDict;
	}
	
	private static double getDocFieldTF(String term, String type, Map<String, Map<String, Double>> tfDoc){
		Map<String, Double> m = tfDoc.get(type);
		if(m != null && m.containsKey(term)){
			return m.get(term);
		}else{
			return 0.0;
=======
		
		private static double getDocFieldTF(String term, String type, Map<String, Map<String, Double>> tfDoc){
			Map<String, Double> m = tfDoc.get(type);
			if(m != null && m.containsKey(term)){
				return m.get(term);
			}else{
				return 0.0;
			}
>>>>>>> FETCH_HEAD
		}

		@Override
		public Classifier training(Instances dataset) {
			try {
				this.model.buildClassifier(dataset);
			} catch (Exception e) {
				e.printStackTrace();
			}
			return this.model;
		}

<<<<<<< HEAD
	@Override
	public TestFeatures extract_test_features(String test_data_file,
			Map<String, Double> idfs) throws Exception {
	
		TestFeatures tf = new TestFeatures("plusPair");
		Instances dataset = null;
		
		/* Build attributes list */
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		attributes.add(new Attribute("url_w"));
		attributes.add(new Attribute("title_w"));
		attributes.add(new Attribute("body_w"));
		attributes.add(new Attribute("header_w"));
		attributes.add(new Attribute("anchor_w"));
		attributes.add(new Attribute("bm25"));
		attributes.add(new Attribute("pagerank"));
		attributes.add(new Attribute("smallestwindow"));
		attributes.add(new Attribute("class",Arrays.asList("-1","+1")));
		dataset = new Instances("test_dataset", attributes, 0);
		
		/* Add data */
		Map<Query,List<Document>> queryDocs = Util.loadTrainData(test_data_file);
		Map<Query,Map<String, Document>> queryDocsBm25 = loadTrainData(test_data_file);
		BM25Scorer bm25Scorer = new BM25Scorer(queryDocsBm25);
		SmallestWindowScorer windowScorer = new SmallestWindowScorer(idfs, queryDocsBm25);
		
		for(Query q: queryDocs.keySet()){
			String query = q.query;
			//System.out.println(query);
			for(Document d: queryDocs.get(q)){
				Map<String,Map<String, Double>> tfDoc = AScorer.getDocTermFreqs(d, q);
				
				
				double tfIdfUrl = 0.0;
				double tfIdfTitle = 0.0;
				double tfIdfBody = 0.0;
				double tfIdfHeader = 0.0;
				double tfIdfAnchor = 0.0;
				double relevanceScore = 0.0;
				double bm25 = bm25Scorer.getSimScore(d, q, idfs);
				double pagerank = d.page_rank;
				double smallwindow = windowScorer.getSimScore(d, q, idfs);
				
				for(String term: q.words){
					double qIDF = Util.IDF(term, idfs);
					double d_tf_url = getDocFieldTF(term, "url", tfDoc);
					tfIdfUrl = tfIdfUrl + qIDF * d_tf_url;
=======
		@Override
		public TestFeatures extract_test_features(String test_data_file,
				Map<String, Double> idfs) throws Exception {
		
			TestFeatures tf = new TestFeatures("pair");
			Instances dataset = null;
			
			/* Build attributes list */
			ArrayList<Attribute> attributes = new ArrayList<Attribute>();
			attributes.add(new Attribute("url_w"));
			attributes.add(new Attribute("title_w"));
			attributes.add(new Attribute("body_w"));
			attributes.add(new Attribute("header_w"));
			attributes.add(new Attribute("anchor_w"));
			attributes.add(new Attribute("class",Arrays.asList("-1","+1")));
			dataset = new Instances("test_dataset", attributes, 0);
			
			/* Add data */
			Map<Query,List<Document>> queryDocs = Util.loadTrainData(test_data_file);
			
			for(Query q: queryDocs.keySet()){
				String query = q.query;
				//System.out.println(query);
				for(Document d: queryDocs.get(q)){
					Map<String,Map<String, Double>> tfDoc = AScorer.getDocTermFreqs(d, q);
>>>>>>> FETCH_HEAD
					
					
					double tfIdfUrl = 0.0;
					double tfIdfTitle = 0.0;
					double tfIdfBody = 0.0;
					double tfIdfHeader = 0.0;
					double tfIdfAnchor = 0.0;
					double relevanceScore = 1.0;
					
					for(String term: q.words){
						double qIDF = Util.IDF(term, idfs);
						double d_tf_url = getDocFieldTF(term, "url", tfDoc);
						tfIdfUrl = tfIdfUrl + qIDF * d_tf_url;
						
						double d_tf_title = getDocFieldTF(term, "title", tfDoc);
						tfIdfTitle = tfIdfTitle + qIDF * d_tf_title;
						
						double d_tf_body = getDocFieldTF(term, "body", tfDoc);
						tfIdfBody = tfIdfBody + qIDF * d_tf_body;
						
						double d_tf_header = getDocFieldTF(term, "header", tfDoc);
						tfIdfHeader = tfIdfHeader + qIDF * d_tf_header;
						
						double d_tf_anchor = getDocFieldTF(term, "anchor", tfDoc);
						tfIdfAnchor = tfIdfAnchor + qIDF * d_tf_anchor;
					}
					
					//Getting Relevance Score
					String url = d.url;
		
					double[] instance = new double[6];
					instance[0] = tfIdfUrl;
					instance[1] = tfIdfTitle;
					instance[2] = tfIdfBody;
					instance[3] = tfIdfHeader;
					instance[4] = tfIdfAnchor;
					instance[5] = relevanceScore;
					dataset.add(new DenseInstance(1.0, instance));
					tf.add(query, url, new DenseInstance(1.0, instance));
				}
<<<<<<< HEAD
				
				//Getting Relevance Score
				String url = d.url;
	
				double[] instance = new double[9];
				instance[0] = tfIdfUrl;
				instance[1] = tfIdfTitle;
				instance[2] = tfIdfBody;
				instance[3] = tfIdfHeader;
				instance[4] = tfIdfAnchor;
				instance[5] = bm25;
				instance[6] = pagerank;
				instance[7] = smallwindow;
				instance[8] = relevanceScore;
				dataset.add(new DenseInstance(1.0, instance));
				tf.add(query, url, new DenseInstance(1.0, instance));
=======
>>>>>>> FETCH_HEAD
			}
			
			//System.out.println(tf.features);
			/* Set last attribute as target */
			dataset.setClassIndex(dataset.numAttributes() - 1);
			Standardize filter = new Standardize();
			filter.setInputFormat(dataset);
			Instances new_dataset = Filter.useFilter(dataset, filter);
			tf.StandardizeFeatures();
			//System.out.println(tf.features);
		
		return tf;
		}

		@Override
		public Map<String, List<String>> testing(TestFeatures tf,
				final Classifier model) {
	      
			Map<String, List<String>> result = new HashMap<String, List<String>>();
			
			for(Map.Entry<String, Map<String, Integer>> entry : tf.index_map.entrySet()) {
				//System.out.println(entry.getKey());
				String query = entry.getKey();
				List<Pair<String,Instance>> urlAndScores = new ArrayList<Pair<String,Instance>>();
				for(Map.Entry<String, Integer> doc : entry.getValue().entrySet()) {
					String url = doc.getKey();
					if(!result.containsKey(query)) {
						result.put(query, new ArrayList<String>());
					}
					urlAndScores.add(new Pair<String,Instance>(url,tf.getInstance(query, url)));
					//TODO compute score and add to map
				}
				
<<<<<<< HEAD
				@Override
				public int compare(Pair<String, Instance> o1, Pair<String, Instance> o2) 
				{
					Instance prev = o1.getSecond();
					Instance curr = o2.getSecond();
					
					//String C = prev.value(5)>curr.value(5) ? "+1" : "-1";
					Instances dataset = null;
					
					/* Build attributes list */
					ArrayList<Attribute> attributes = new ArrayList<Attribute>();
					attributes.add(new Attribute("url_w"));
					attributes.add(new Attribute("title_w"));
					attributes.add(new Attribute("body_w"));
					attributes.add(new Attribute("header_w"));
					attributes.add(new Attribute("anchor_w"));
					attributes.add(new Attribute("bm25"));
					attributes.add(new Attribute("pagerank"));
					attributes.add(new Attribute("smallestwindow"));
					attributes.add(new Attribute("class",Arrays.asList("-1","+1")));
					dataset = new Instances("merge_dataset", attributes, 0);
					
					double url = prev.value(0)-curr.value(0);
					double title = prev.value(1)-curr.value(1);
					double body = prev.value(2)-curr.value(2);
					double header = prev.value(3)-curr.value(3);
					double anchor = prev.value(4)-curr.value(4);
					double bm25 = prev.value(5)-curr.value(4);
					double pageRank = prev.value(6)-curr.value(4);
					double window = prev.value(7)-curr.value(4);
					Instance merge = new DenseInstance(9);
					merge.setDataset(dataset);
					merge.setValue(0, url);
					merge.setValue(1, title);
					merge.setValue(2, body);
					merge.setValue(3, header);
					merge.setValue(4, anchor);
					merge.setValue(5, bm25);
					merge.setValue(6, pageRank);
					merge.setValue(7, window);
					dataset.add(merge);
					dataset.setClassIndex(dataset.numAttributes() - 1);
					//System.out.println("before merged: "+merge);
					double score = 0;
					try {
						score = model.classifyInstance(dataset.get(0));
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
=======
				//sort urls for query based on scores
				Collections.sort(urlAndScores, new Comparator <Pair<String,Instance>>() {
					
					@Override
					public int compare(Pair<String, Instance> o1, Pair<String, Instance> o2) 
					{
						Instance prev = o1.getSecond();
						Instance curr = o2.getSecond();
						
						//String C = prev.value(5)>curr.value(5) ? "+1" : "-1";
						Instances dataset = null;
						
						/* Build attributes list */
						ArrayList<Attribute> attributes = new ArrayList<Attribute>();
						attributes.add(new Attribute("url_w"));
						attributes.add(new Attribute("title_w"));
						attributes.add(new Attribute("body_w"));
						attributes.add(new Attribute("header_w"));
						attributes.add(new Attribute("anchor_w"));
						attributes.add(new Attribute("class",Arrays.asList("-1","+1")));
						dataset = new Instances("merge_dataset", attributes, 0);
						
						double url = prev.value(0)-curr.value(0);
						double title = prev.value(1)-curr.value(1);
						double body = prev.value(2)-curr.value(2);
						double header = prev.value(3)-curr.value(3);
						double anchor = prev.value(4)-curr.value(4);
						Instance merge = new DenseInstance(6);
						merge.setDataset(dataset);
						merge.setValue(0, url);
						merge.setValue(1, title);
						merge.setValue(2, body);
						merge.setValue(3, header);
						merge.setValue(4, anchor);
						//merge.setValue(5, "-1");
						dataset.add(merge);
						dataset.setClassIndex(dataset.numAttributes() - 1);
						//System.out.println("before merged: "+merge);
						double score = 0;
						try {
							score = model.classifyInstance(dataset.get(0));
						} catch (Exception e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						//System.out.println(score>0.0);
						//System.out.println("after merged: "+(score>0.0));
						return (score>0.0)?-1:1;
						
					}	
				});
				
				for (Pair<String,Instance> urlAndScore : urlAndScores) {
					//System.out.println("\turl: "+urlAndScore.getFirst());
					result.get(query).add(urlAndScore.getFirst());
>>>>>>> FETCH_HEAD
					}
			}
			
			return result;
		}
		
}
