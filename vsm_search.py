import os
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# --- NLTK Setup ---
# You may need to run these downloads once if you don't have them
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
# --- END NLTK Setup ---


class VectorSpaceModel:
    """
    Implements a Vector Space Model for information retrieval.
    Follows the lnc.ltc weighting scheme.
    """

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Core data structures
        self.doc_filenames = []
        self.doc_lengths = defaultdict(float)
        self.inverted_index = defaultdict(list)
        self.doc_freq = defaultdict(int)
        
        # Build the index upon initialization
        self._build_index()

    def _preprocess(self, text):
        """
        Preprocesses text: tokenization, lowercasing, stopword removal, stemming.
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return stemmed_tokens

    def _build_index(self):
        """
        Builds the inverted index, document frequency, and document lengths from the corpus.
        """
        print("Building index...")
        files = [f for f in os.listdir(self.corpus_path) if f.endswith('.txt')]
        self.doc_filenames = sorted(files)
        doc_id_counter = 0

        # First pass: build inverted index and tf for each doc
        for filename in self.doc_filenames:
            doc_id = doc_id_counter
            filepath = os.path.join(self.corpus_path, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tokens = self._preprocess(content)
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1

            # Store term frequencies in the inverted index
            for term, tf in term_freq.items():
                self.inverted_index[term].append((doc_id, tf))

            # Calculate and store document lengths (lnc part)
            sum_of_squares = 0.0
            for tf in term_freq.values():
                weight = 1 + math.log10(tf)
                sum_of_squares += weight ** 2
            self.doc_lengths[doc_id] = math.sqrt(sum_of_squares)
            
            doc_id_counter += 1

        # Second pass: calculate document frequency (df) for each term
        for term, postings in self.inverted_index.items():
            self.doc_freq[term] = len(postings)

        print(f"Index built successfully. {len(self.doc_filenames)} documents processed.")

    def search(self, query_text):
        """
        Performs a search for a given query and returns ranked documents.
        """
        if not self.doc_filenames:
            print("Index is empty. Cannot search.")
            return []

        query_tokens = self._preprocess(query_text)
        
        # Calculate query vector (ltc part)
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1

        query_vector = defaultdict(float)
        query_length = 0.0
        num_docs = len(self.doc_filenames)

        for term, tf in query_tf.items():
            if term in self.doc_freq:
                # Calculate tf-idf weight for query term
                idf = math.log10(num_docs / self.doc_freq[term])
                weight = (1 + math.log10(tf)) * idf
                query_vector[term] = weight
                query_length += weight ** 2
        
        # Normalize the query vector
        query_length = math.sqrt(query_length)
        if query_length > 0:
            for term in query_vector:
                query_vector[term] /= query_length

        # --- Score Calculation ---
        scores = defaultdict(float)
        for query_term, query_weight in query_vector.items():
            if query_term in self.inverted_index:
                for doc_id, doc_tf in self.inverted_index[query_term]:
                    # Document term weight (lnc) is (1 + log(tf))
                    doc_term_weight = 1 + math.log10(doc_tf)
                    
                    # Accumulate score (dot product)
                    scores[doc_id] += query_weight * doc_term_weight

        # Normalize scores by document lengths
        for doc_id in scores:
            if self.doc_lengths[doc_id] > 0:
                scores[doc_id] /= self.doc_lengths[doc_id]
        
        # --- Ranking ---
        # Sort by score (desc), then by docID (asc) as a tie-breaker
        ranked_docs = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        # Format results with filenames
        results = [(self.doc_filenames[doc_id], score) for doc_id, score in ranked_docs if score > 0]
        
        return results[:10]

def soundex(name):
    """
    Implements the Soundex algorithm for phonetic matching of names.
    This is the "novelty" part of the assignment.
    """
    if not name:
        return ""

    # 1. Retain the first letter and convert to uppercase
    first_letter = name[0].upper()
    name = name.lower()
    
    # 2. Map letters to numbers
    s_map = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    
    # 3. Build the code
    code = first_letter
    last_code = s_map.get(name[0], '0')
    
    for char in name[1:]:
        digit = s_map.get(char, '0')
        if digit != '0' and digit != last_code:
            code += digit
            last_code = digit
        elif char in 'aeiouyhw':
            last_code = '0' # Vowels separate consonants

    # 4. Pad with zeros and truncate
    code = code.replace('0', '')
    code += '000'
    return code[:4]

# --- Main Execution ---
if __name__ == "__main__":
    # IMPORTANT: Create a folder named 'corpus' in the same directory as this script
    # and place all your .txt document files inside it.
    CORPUS_FOLDER = 'corpus'
    
    if not os.path.exists(CORPUS_FOLDER):
        print(f"Error: The '{CORPUS_FOLDER}' directory was not found.")
        print("Please create it and add your text files to it.")
    else:
        vsm = VectorSpaceModel(CORPUS_FOLDER)
        
        # Example usage of Soundex
        print("\n--- Soundex Example ---")
        name1 = "Robert"
        name2 = "Rupert"
        print(f"Soundex for '{name1}': {soundex(name1)}")
        print(f"Soundex for '{name2}': {soundex(name2)}")
        print("-----------------------\n")

        # Interactive search loop
        print("Enter a query to search, or type 'exit' to quit.")
        while True:
            user_query = input("query> ")
            if user_query.lower() == 'exit':
                break
            
            results = vsm.search(user_query)
            
            if not results:
                print("No relevant documents found.")
            else:
                print("\nTop 10 Search Results:")
                for i, (filename, score) in enumerate(results):
                    print(f"{i+1}. ('{filename}', {score})")
            print("-" * 20)
