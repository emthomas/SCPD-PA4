package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Document {
	public String url = null;
	public String title = null;
	public List<String> headers = null;
	public Map<String, List<Integer>> body_hits = null; // term -> [list of positions]
	public int body_length = 0;
	public int page_rank = 0;
	public Map<String, Integer> anchors = null; // term -> anchor_count

	public Document(){
	}
	
	public Document(String url){
		this.url=url;
	}
	
	// For debug
	public String toString() {
		StringBuilder result = new StringBuilder();
		String NEW_LINE = System.getProperty("line.separator");
		if (title != null) result.append("title: " + title + NEW_LINE);
		if (headers != null) result.append("headers: " + headers.toString() + NEW_LINE);
		if (body_hits != null) result.append("body_hits: " + body_hits.toString() + NEW_LINE);
		if (body_length != 0) result.append("body_length: " + body_length + NEW_LINE);
		if (page_rank != 0) result.append("page_rank: " + page_rank + NEW_LINE);
		if (anchors != null) result.append("anchors: " + anchors.toString() + NEW_LINE);
		return result.toString();
	}
	
	//Implementing here with two List<String>
	//Should be easy enough to use arrays, or streams, or whatever.
	public int getSmallestWindow(Query q) {
	    int minDistance = Integer.MAX_VALUE-1;
	    List<String> words = q.words;
	    //Create a map of the last known position of each word
	    Map<String, Integer> map = new HashMap();
	    Map<String, Integer> index = new HashMap();
	    for (String word : words) {
	    	if(body_hits==null || !body_hits.containsKey(word)) {
				return minDistance; //doesn't contain all terms
			}
	        map.put(word, body_hits.get(word).get(0));
	        index.put(word, 0);
	    }
	    
	    String word = null;
	    boolean done = false;
	    
	    //Loop through body hits
	    while(!done){
	            int curDistance = getCurDistance(map);
	            if (curDistance < minDistance)
	                minDistance = curDistance;
	            
	            String minWord = getMinWord(map);
	            int currPos = map.get(minWord);
	            int currIndex = body_hits.get(minWord).indexOf(currPos);
	            
	            if(currIndex+1==body_hits.get(minWord).size()) {
	            	done=true;
	            	break;
	            }
	            
	            int newPos = body_hits.get(minWord).get(currIndex+1);
	            map.put(minWord, newPos);
	    }
	    
	    //look for min Distance in title
	    if (title != null) {
	    	List<String> text = new ArrayList<String>(Arrays.asList(title.toLowerCase().split(" ")));
	    	int curDistance = getShortestSubseqWith(text,words);
	    	if (curDistance < minDistance)
                minDistance = curDistance;
	    }
	    
	  //look for min Distance in headers
		if (headers != null) {
			for(String header : headers) {
				List<String> text = new ArrayList<String>(Arrays.asList(header.toLowerCase().split(" ")));
		    	int curDistance = getShortestSubseqWith(text,words);
		    	if (curDistance < minDistance)
	                minDistance = curDistance;
			}
		}
		
		//look for min Distance in anchors
		if (anchors != null){
			for(String anchor : anchors.keySet()) {
				List<String> text = new ArrayList<String>(Arrays.asList(anchor.toLowerCase().split(" ")));
		    	int curDistance = getShortestSubseqWith(text,words);
		    	if (curDistance < minDistance)
	                minDistance = curDistance;
			}
		}
	    
	    return minDistance+1;
	}
	
	public static int getShortestSubseqWith(List<String> text, List<String> words) {
	    int minDistance = Integer.MAX_VALUE;
	    //Create a map of the last known position of each word
	    Map<String, Integer> map = new HashMap();
	    for (String word : words) {
	        map.put(word, -1);
	    }
	    String word;
	    //One loop through the main search string
	    for (int position = 0; position < text.size(); position++){
	        word = text.get(position);
	        //If the current word found is in the list we're looking for
	        if (map.containsKey(word)) {
	            //Update the map
	            map.put(word, position);
	            //And if the current positions are the closest seen so far, update the min value.
	            int curDistance = getCurDistance(map);
	            if (curDistance < minDistance)
	                minDistance = curDistance;
	        }
	    }
	    return minDistance;
	}

	//Get the current distance between the last known position of each value in the map
	private static int getCurDistance(Map<String, Integer> map) {
	    int min = Integer.MAX_VALUE;
	    int max = 0;
	    for (Integer value : map.values()) {
	        if (value == -1)
	            return Integer.MAX_VALUE;
	        else {
	            max = Math.max(max,value);
	            min = Math.min(min,value);
	        }
	    }
	    return max - min;
	}
	
	private static String getMinWord(Map<String, Integer> map) {
	    int min = Integer.MAX_VALUE;
	    for (Integer value : map.values()) {
	            min = Math.min(min,value);
	        }
	    
	    for (Map.Entry<String, Integer> value : map.entrySet()) {
            if(value.getValue()==min) {
            	return value.getKey();
            }
        }
	    return "";
	}
}
