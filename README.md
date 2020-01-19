
The identification of topic diversity in text conversations is becoming increasingly relevant to a variety of applications, 
including content summarization and keyword extraction. This project introduce a simple text analysis method that focuses on 
identifying the key concepts and detecting topic diversity using network analysis. The text is first represented as a 
directed network using synonym relationships. Then community detection and other measures will be used to identify the key 
concepts within the text. Finally, sentiment analysis is conducted to identify the general orientation of the introduced 
opinions and assess topic diversity.

Run from terminal: $ python3 Topic-Diversity.py filename  m n

Input: - a conversation (set of text items such as replies to a tweet or comments to a news feed) in a text file form.
       - m: the number of key(central) concepts to be included in the analysis.
       - n: the minimum frequency that a singleton node must achieve to be included in the se of key concepts.

Output: Text files: text graph & list of nodes with their frequencies
        On  screen: list of communities with their properties & the sentiment analysis of the key words 
        
        
