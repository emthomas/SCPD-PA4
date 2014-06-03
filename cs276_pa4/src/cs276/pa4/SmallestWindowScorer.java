package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class SmallestWindowScorer extends BM25Scorer{

	/////smallest window specifichyperparameters////////
    double B = 10;    	    
    double boostmod = 1;
    Map<Document,Map<Query,Double>> smallestWindowDict = null;
    
    //////////////////////////////
    
	public SmallestWindowScorer(Map<String, Double> idfs,Map<Query,Map<String, Document>> queryDict) 
	{
		super(idfs, queryDict);
		handleSmallestWindow();
	}

	
	public void handleSmallestWindow()
	{
		smallestWindowDict = new HashMap<Document,Map<Query,Double>>();
		
		for(Query q : queryDict.keySet()) {
			for(String url : queryDict.get(q).keySet()) {
				Document doc = queryDict.get(q).get(url);
				double minWindow = (double)doc.getSmallestWindow(q);
				Map<Query,Double> queryWin = new HashMap<Query,Double>();
				queryWin.put(q, minWindow);
				smallestWindowDict.put(doc, queryWin);
			}
		}
	}
	
	public double getBoost(Document d, Query q) {
		boostmod = smallestWindowDict.get(d).get(q)/Integer.MAX_VALUE;
		double window = smallestWindowDict.get(d).get(q);
		if(window==Integer.MAX_VALUE) {
			boostmod = 1;
		}
		else if(window==q.getUniqueTermsCount()) {
			boostmod = B;
		}
		else {
			boostmod=B*Math.pow(Math.E,-window);
		}
		
		return boostmod;
	}
	
	public double getSimScore(Document d, Query q,  Map<String, Double> dfs) {
		Map<String,Map<String, Double>> tfs = AScorer.getDocTermFreqs(d,q);
		
		this.normalizeTFs(tfs, d, q);
		
		Map<String,Double> tfQuery = AScorer.getQueryFreqs(q);
		
		return getNetScore(tfs,q,tfQuery,d,dfs)*getBoost(d,q);
	}

}
