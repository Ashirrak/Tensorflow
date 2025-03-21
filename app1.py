import streamlit as st
import time  
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import graphviz
import seaborn as sns
def compute_prefix_function(pattern):
    m = len(pattern)
    B = [0] * m  
    j = 0  

    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = B[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        B[i] = j
    
    return B

def morris_pratt_search(text, pattern):
    n, m = len(text), len(pattern)
    B = compute_prefix_function(pattern)
    j = 0  
    occurrences = []
    comparisons = 0
    
    for i in range(n):
        comparisons += 1
        while j > 0 and text[i] != pattern[j]:
            j = B[j - 1]
            comparisons += 1
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            occurrences.append(i - m + 1)
            j = B[j - 1]
    
    return B, comparisons, occurrences

def bad_character_table(pattern):
    m = len(pattern)
    bad_char = {char: m for char in set(pattern)}
    for i in range(m - 1):
        bad_char[pattern[i]] = m - 1 - i
    return bad_char

def boyer_moore_search(text, pattern):
    n, m = len(text), len(pattern)
    bad_char = bad_character_table(pattern)
    occurrences = []
    comparisons = 0
    i = 0  

    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            comparisons += 1
            j -= 1
        if j < 0:
            occurrences.append(i)
            i += bad_char.get(text[i + m], m) if i + m < n else 1
        else:
            comparisons += 1
            i += max(1, bad_char.get(text[i + j], m) - (m - 1 - j))
    
    return bad_char, comparisons, occurrences


def hash_function(pattern, base=256, prime=101):
    hash_value = 0
    for char in pattern:
        hash_value = (hash_value * base + ord(char)) % prime
    return hash_value
    
def rabin_karp_multiple_search(text, patterns, base=256, prime=101):
    n = len(text)
    results = {p: [] for p in patterns}
    pattern_hashes = {p: hash_function(p, base, prime) for p in patterns}
    comparisons = 0

    for pattern in patterns:
        m = len(pattern)
        if m > n:  
            continue

        text_hash = hash_function(text[:m], base, prime)
        high_order = pow(base, m-1, prime)

        for i in range(n - m + 1):
            comparisons += 1
            if text_hash == pattern_hashes[pattern]:
                if text[i:i + m] == pattern:
                    results[pattern].append(i)

            if i < n - m:
                text_hash = (text_hash - ord(text[i]) * high_order) * base + ord(text[i + m])
                text_hash = text_hash % prime

    return results, comparisons


def rabin_karp_bloom_filter(text, patterns, hash_functions):
    all_results = {p: [] for p in patterns}
    all_comparisons = 0
    n = len(text)
    false_positives = {}

    for pattern in patterns:
        m = len(pattern)
        if m > n:
            continue

        pattern_hashes = {h: hash_function(pattern, h[0], h[1]) for h in hash_functions}
        text_hashes = {h: hash_function(text[:m], h[0], h[1]) for h in hash_functions}
        high_orders = {h: pow(h[0], m - 1, h[1]) for h in hash_functions}

        for i in range(n - m + 1):
            all_comparisons += 1
            substring = text[i:i + m]

            if all(text_hashes[h] == pattern_hashes[h] for h in hash_functions):
                if substring == pattern:
                    all_results[pattern].append(i)
                else:
                    false_positives.setdefault(pattern, []).append((substring, i))  

            if i < n - m:
                for h in hash_functions:
                    text_hashes[h] = (text_hashes[h] - ord(text[i]) * high_orders[h]) * h[0] + ord(text[i + m])
                    text_hashes[h] = text_hashes[h] % h[1]

    return all_results, all_comparisons, false_positives



class TrieNode:
    """A class representing a node in the Trie."""
    def __init__(self):
        self.children = {}
        self.output = set()
        self.fail = None

class AhoCorasick:
    def __init__(self):
        self.root = TrieNode()
        self.nodes = [self.root]  # Store all nodes for tracking
        self.node_ids = {self.root: "Root"}  # Assign string IDs

    def insert(self, word, index):
        """Insert a word into the Trie."""
        node = self.root
        for char in word:
            if char not in node.children:
                new_node = TrieNode()
                node.children[char] = new_node
                self.nodes.append(new_node)
                self.node_ids[new_node] = f"N{len(self.nodes)}"
            node = node.children[char]
        node.output.add(word)  # Mark the end of a pattern

    def build(self):
        """Build the failure function using BFS."""
        queue = deque()

        # Set failure links for root's children
        for char, child in self.root.children.items():
            child.fail = self.root
            queue.append(child)

        # BFS to set failure links for other nodes
        while queue:
            current = queue.popleft()
            for char, child in current.children.items():
                queue.append(child)

                # Set failure function
                fail_node = current.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail

                child.fail = fail_node.children[char] if fail_node else self.root
                child.output |= child.fail.output  # Merge outputs

    def visualize_trie(self):
        """Generate Graphviz representation of the Trie."""
        graph = graphviz.Digraph()
        graph.node("Root", shape="circle", color="red")

        for node in self.nodes:
            node_id = self.node_ids[node]
            if node.output:
                graph.node(node_id, shape="doublecircle", color="blue")
            else:
                graph.node(node_id, shape="circle")

            for char, child in node.children.items():
                child_id = self.node_ids[child]
                graph.edge(node_id, child_id, label=char)

        return graph

    def visualize_search_path(self, text):
        """Generate Graphviz representation of the search process."""
        graph = graphviz.Digraph()
        node = self.root
        graph.node("Start", shape="circle", color="green")

        for i, char in enumerate(text):
            if node is not None and char in node.children:
                node = node.children[char]
            else:
                while node is not None and char not in node.children:
                    node = node.fail
                node = node.children[char] if node else self.root
            
            if node:
                graph.node(f"Step {i+1}", label=f"'{char}'", shape="circle", color="orange")
                graph.edge(f"Step {i}", f"Step {i+1}", label=f"Match: {char}")

        return graph

    def search(self, text):
        """Search text using the Aho-Corasick automaton."""
        node = self.root
        results = []
        comparisons = 0

        for i, char in enumerate(text):
            comparisons += 1
            while node is not None and char not in node.children:
                node = node.fail  # Follow failure links
            
            if node is None:
                node = self.root
                continue

            node = node.children[char]
            if node.output:
                for pattern in node.output:
                    results.append((i - len(pattern) + 1, pattern))

        return results, comparisons

class CommentzWalter:
    def __init__(self, patterns):
        self.patterns = patterns
        self.bad_char_table = {}
        self.shift_table = {}
        self.build_tables()

    def build_tables(self):
        # Build bad character table
        for pattern in self.patterns:
            length = len(pattern)
            for i in range(length - 1):
                self.bad_char_table[(pattern, pattern[i])] = length - i - 1

        # Build shift table
        for pattern in self.patterns:
            length = len(pattern)
            for i in range(length):
                suffix = pattern[i:]
                shift = length - i
                for j in range(len(self.patterns)):
                    if self.patterns[j].endswith(suffix) and self.patterns[j] != pattern:
                        shift = min(shift, length - len(self.patterns[j]))
                self.shift_table[(pattern, suffix)] = shift

    def search(self, text):
        results = {pattern: [] for pattern in self.patterns}  # Store occurrences per pattern
        comparisons = 0

        for pattern in self.patterns:
            m, n = len(pattern), len(text)
            i = 0
            while i <= n - m:
                j = m - 1
                while j >= 0 and i + j < n and pattern[j] == text[i + j]:
                    j -= 1
                    comparisons += 1
                if j < 0:  # Match found
                    results[pattern].append(i)
                    i += self.shift_table.get((pattern, ""), m) if (pattern, "") in self.shift_table else m
                else:
                    bad_char_shift = self.bad_char_table.get((pattern, text[i + j]), m) if i + j < n else m
                    i += max(1, bad_char_shift)

        return results, comparisons


algorithm = st.sidebar.radio("Algorithme", (
    "Morris-Pratt", 
    "Boyer-Moore", 
    "Rabin-Karp (1 fonction de hachage)", 
    "Rabin-Karp (3 fonctions de hachage)",
    "Test",
    "Aho-Corasick Multiple Pattern Matching",
    "Commentz-Walter",
))

# Streamlit UI
if algorithm != "Test":
    st.title("Algorithmes de Recherche de Motif")
else:
    st.title("Tests et Analyses")
    
if algorithm != "Test":
    text = st.text_area("Entrez le texte", "ababcababcabc")

    
    if "Rabin-Karp" in algorithm:
        patterns_input = st.text_area("Entrez les motifs pour Rabin-Karp (séparés par une virgule)", "abc, ab")
        patterns = [p.strip() for p in patterns_input.split(",") if p.strip()]
    elif "Aho-Corasick Multiple Pattern Matching" in algorithm:
        patterns_input = st.text_area("Enter patterns (comma-separated):","abc")
        patterns = [p.strip() for p in patterns_input.split(',') if p.strip()]
    elif "Commentz-Walter" in algorithm:
        patterns_input = st.text_area("Enter patterns (comma-separated)","abc").strip()
        patterns = [p.strip() for p in patterns_input.split(",") if p.strip()]  # Remove empty patterns
    elif algorithm != "Test":
        pattern = st.text_input("Entrez le motif", "abc")
    

# Sélection des fonctions de hachage pour le filtre de Bloom
if algorithm == "Rabin-Karp (3 fonctions de hachage)":
    st.sidebar.subheader("Filtre de Bloom - Sélection des fonctions de hachage")
    bloom_choice = st.sidebar.radio(
        "Choix des fonctions de hachage",
        ["Automatique (3 fonctions par défaut)", "Manuelle (Définir ses propres paramètres)"]
    )

    if bloom_choice == "Manuelle (Définir ses propres paramètres)":
        st.sidebar.write("Définissez vos propres fonctions de hachage:")
        
        hash_functions = []
        for i in range(3):  
            base = st.sidebar.number_input(f"Base {i+1} (ex: 256)", min_value=2, value=256)
            prime = st.sidebar.number_input(f"Prime {i+1} (ex: 101)", min_value=2, value=101)
            hash_functions.append((base, prime))
    else:
        hash_functions = [(256, 101), (257, 103), (259, 107)]  

if algorithm != "Test":
    if st.button("Rechercher"):
        if algorithm == "Morris-Pratt":
            B, comparisons, occurrences = morris_pratt_search(text, pattern)
            st.write("**Tableau des bords:**", B)
            st.write("**Nombre de comparaisons:**", comparisons)
            st.write("**Positions des occurrences:**", occurrences)
        elif algorithm == "Boyer-Moore":
            bad_char, comparisons, occurrences = boyer_moore_search(text, pattern)
            st.write("**Tableau dictionnaire:**", bad_char)
            st.write("**Nombre de comparaisons:**", comparisons)
            st.write("**Positions des occurrences:**", occurrences)
        elif algorithm == "Rabin-Karp (1 fonction de hachage)":
            start = time.time()
            results, comparisons = rabin_karp_multiple_search(text, patterns)
            end = time.time()
            st.write("**Résultats de Rabin-Karp (1 fonction de hachage) :**")
            st.write(f"**Nombre de Comparaisons :** {comparisons}")
            st.write(f"**Temps d'exécution :** {end - start:.6f} secondes")
            for pattern, occurrences in results.items():
                st.write(f"Motif `{pattern}` trouvé aux positions : {occurrences if occurrences else 'Aucune occurrence trouvée'}")
        elif algorithm == "Rabin-Karp (3 fonctions de hachage)":
            start = time.time()
            results, comparisons, false_positives = rabin_karp_bloom_filter(text, patterns, hash_functions)
            end = time.time()
            
            st.write("**Rabin-Karp avec 3 fonctions de hachage (Filtre de Bloom) :**")
            st.write(f"**Total Comparaisons:** {comparisons}")
            st.write(f"**Temps d'exécution:** {end - start:.6f} secondes")
        
            # Display found motifs
            for pattern, occurrences in results.items():
                if occurrences:
                    st.write(f" Motif `{pattern}` trouvé aux positions : {occurrences}")
                else:
                    st.write(f"Motif `{pattern}` non trouvé.")
        
            #  Display false positives
            if false_positives:
                st.write("###  Faux Positifs Détectés:")
                false_pos_list = []
                for pattern, fp_list in false_positives.items():
                    for motif, pos in fp_list:
                        false_pos_list.append({"Motif Original": pattern, "Faux Motif": motif, "Position": pos})
        
                df_false = pd.DataFrame(false_pos_list)
                st.dataframe(df_false)
            else:
                st.write("Aucun faux positif détecté.")

        elif algorithm == "Aho-Corasick Multiple Pattern Matching":
            if not text or not patterns:
                st.warning("Please enter both text and patterns.")
            else:
                # Initialize Aho-Corasick Automaton
                ac = AhoCorasick()
                for pattern in patterns:
                    ac.insert(pattern, len(pattern))
                
                ac.build()
        
                # Measure execution time
                start_time = time.time()
                results, comparisons = ac.search(text)
                end_time = time.time()
        
                # Display results in a table format
                st.subheader("Results")
                if results:
                    results_df = pd.DataFrame(results, columns=["Position", "Pattern"])
                    st.dataframe(results_df)
                else:
                    st.write("No occurrences found.")
        
                st.write(f"Total comparisons: {comparisons}")
                st.write(f"Execution time: {end_time - start_time:.6f} seconds")
        
                # Display Trie Automaton
                st.subheader("Automaton of Prefixes (Trie Structure)")
                st.graphviz_chart(ac.visualize_trie())
        
                # Display Search Path Automaton
                st.subheader("Search Path Automaton (Pattern Matching Process)")
                st.graphviz_chart(ac.visualize_search_path(text))

        elif algorithm == "Commentz-Walter":
            if not patterns:
                st.warning("Please enter at least one pattern.")
            else:
                # Continue with processing
                cw = CommentzWalter(patterns)  # Instantiate the CW class
                start = time.time()
                results, comparisons = cw.search(text)  # Perform search
                end = time.time()
            
                st.write("**Commentz-Walter Algorithm:**")
                st.write(f"**Total Comparisons:** {comparisons}")
                st.write(f"**Execution Time:** {end - start:.6f} seconds")
            
                # Display found patterns
                for pattern, occurrences in results.items():
                    if occurrences:
                        st.write(f"Motif `{pattern}` trouvé aux positions : {occurrences}")
                    else:
                        st.write(f"Motif `{pattern}` non trouvé.")
            
                


if algorithm == "Test":
    
    test_choice = st.sidebar.radio(
        "Choix de tests",
        ["Test Morris-Pratt et Boyer-Moore", "Test Rabin-Karp (1 vs 3 fonctions de hachage)","Test Aho-Corasick","Test Commentz-Walter"]
    )
    if test_choice == "Test Morris-Pratt et Boyer-Moore":
    # ===================== Test et analyse de MP et BM =====================
        st.subheader("Test Morris-Pratt et Boyer-Moore")
        text_mp_bm = st.text_area("Entrez le texte pour MP/BM", "abababababababababab")
        pattern_mp_bm = st.text_input("Entrez le motif pour MP/BM", "aba")
        
        if st.button("Exécuter MP & BM"):
            if text_mp_bm and pattern_mp_bm:
                results = []
        
                # Test Morris-Pratt
                start_mp = time.time()
                _, comparisons_mp, _ = morris_pratt_search(text_mp_bm, pattern_mp_bm)
                end_mp = time.time()
        
                # Test Boyer-Moore
                start_bm = time.time()
                _, comparisons_bm, _ = boyer_moore_search(text_mp_bm, pattern_mp_bm)
                end_bm = time.time()
        
                results.append({
                    "Taille du Texte": len(text_mp_bm),
                    "Taille du Motif": len(pattern_mp_bm),
                    "Comparaisons MP": comparisons_mp,
                    "Temps MP (s)": round(end_mp - start_mp, 6),
                    "Comparaisons BM": comparisons_bm,
                    "Temps BM (s)": round(end_bm - start_bm, 6),
                })
            
                df = pd.DataFrame(results)
                st.write("### Comparaison Morris-Pratt vs Boyer-Moore")
                st.dataframe(df)
            else:
                st.warning("Veuillez entrer un texte et un motif.")
    elif test_choice == "Test Rabin-Karp (1 vs 3 fonctions de hachage)":
        st.subheader("Comparaison des performances entre RK1 et RK3")

        text_rk = st.text_area("Entrez le texte pour RK", "abababababababababab")
        pattern_rk = st.text_input("Entrez le motif pour RK", "aba")
        num_filters = st.slider("Nombre de filtres Bloom pour RK3", 1, 5, 3)
        
        if st.button("Exécuter RK1 & RK3"):
            if text_rk and pattern_rk:
                hash_functions_set = [(50 + i, 31 + i * 2) for i in range(num_filters)]
        
                # Execute RK1
                start_rk1 = time.time()
                results_rk1, comparisons_rk1 = rabin_karp_multiple_search(text_rk, [pattern_rk])
                end_rk1 = time.time()
        
                # Execute RK3 with false positives tracking
                start_rk3 = time.time()
                results_rk3, comparisons_rk3, false_positives = rabin_karp_bloom_filter(text_rk, [pattern_rk], hash_functions_set)
                end_rk3 = time.time()
        
                # Compute false positive count
                false_positive_count = len(false_positives.get(pattern_rk, []))
        
                # Main results table
                results = [{
                    "Méthode": "RK1",
                    "Comparaisons": comparisons_rk1,
                    "Temps (s)": round(end_rk1 - start_rk1, 6),
                    "Faux Positifs": 0
                }, {
                    "Méthode": "RK3",
                    "Comparaisons": comparisons_rk3,
                    "Temps (s)": round(end_rk3 - start_rk3, 6),
                    "Faux Positifs": false_positive_count
                }]
        
                df = pd.DataFrame(results)
                st.write("### Résultats de la comparaison")
                st.dataframe(df)
        
                # Display motifs found and positions
                motifs_results = [{
                    "Méthode": "RK1",
                    "Positions trouvées": results_rk1[pattern_rk],
                    "Nombre de motifs trouvés": len(results_rk1[pattern_rk])
                }, {
                    "Méthode": "RK3",
                    "Positions trouvées": results_rk3[pattern_rk],
                    "Nombre de motifs trouvés": len(results_rk3[pattern_rk])
                }]
        
                df_motifs = pd.DataFrame(motifs_results)
                st.write("### Motifs trouvés et leurs positions")
                st.dataframe(df_motifs)
        
                # Display false positive motifs
                if false_positives:
                    false_pos_list = [{
                        "Motif Original": pattern_rk,
                        "Faux Motif": motif,
                        "Position": pos
                    } for motif, pos in false_positives.get(pattern_rk, [])]
        
                    df_false = pd.DataFrame(false_pos_list)
                    st.write("### Faux Positifs Trouvés")
                    st.dataframe(df_false)
        
                # Visualizations
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sns.barplot(x="Méthode", y="Comparaisons", data=df, ax=axes[0])
                axes[0].set_title("Nombre de Comparaisons")
        
                sns.barplot(x="Méthode", y="Temps (s)", data=df, ax=axes[1])
                axes[1].set_title("Temps d'Exécution")
        
                st.pyplot(fig)
        
                # False positives visualization
                fig_fp, ax_fp = plt.subplots(figsize=(6, 4))
                sns.barplot(x=["RK1", "RK3"], y=[0, false_positive_count], ax=ax_fp)
                ax_fp.set_title("Nombre de Faux Positifs")
                st.pyplot(fig_fp)
        
            else:
                st.warning("Veuillez entrer un texte et un motif.")
    elif test_choice == "Test Aho-Corasick":
            st.subheader("Test et analyse de l’algorithme Aho-Corasick")
            text_ac = st.text_area("Entrez le texte pour AC", "abababababababababab")
            patterns_ac = st.text_area("Entrez les motifs (séparés par des virgules)", "aba, bab")
            
            if st.button("Exécuter AC"):
                    if text_ac and patterns_ac:
                        patterns_list = [p.strip() for p in patterns_ac.split(",")]
            
                        # Initialize Aho-Corasick Automaton
                        ac = AhoCorasick()
                        for pattern in patterns_list:
                            ac.insert(pattern, len(pattern))
                        
                        ac.build()
            
                        # Measure execution time
                        start_ac = time.time()
                        results_ac, comparisons_ac = ac.search(text_ac)
                        end_ac = time.time()
            
                        results = [{
                            "Taille du Texte": len(text_ac),
                            "Nombre de Motifs": len(patterns_list),
                            "Comparaisons AC": comparisons_ac,
                            "Temps AC (s)": round(end_ac - start_ac, 6)
                        }]
            
                        df = pd.DataFrame(results)
                        st.write("### Résultats de Aho-Corasick")
                        st.dataframe(df)
            
                        # Visualization
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=["Aho-Corasick"], y=[comparisons_ac], ax=ax)
                        ax.set_title("Nombre de Comparaisons - Aho-Corasick")
                        st.pyplot(fig)
            
                    else:
                        st.warning("Veuillez entrer un texte et des motifs.")
        
    elif test_choice == "Test Commentz-Walter":
        st.subheader("Test et analyse de l’algorithme Commentz-Walter")
        text_cw = st.text_area("Entrez le texte pour CW", "abababababababababab")
        patterns_cw = st.text_area("Entrez les motifs (séparés par des virgules)", "aba, bab")
    
        if st.button("Exécuter CW"):
            if text_cw and patterns_cw:
                patterns_list_cw = [p.strip() for p in patterns_cw.split(",")]
                
                cw = CommentzWalter(patterns_list_cw)
                start_cw = time.time()
                results_cw, comparisons_cw = cw.search(text_cw)
                end_cw = time.time()
                
                results = [{
                    "Taille du Texte": len(text_cw),
                    "Nombre de Motifs": len(patterns_list_cw),
                    "Comparaisons CW": comparisons_cw,
                    "Temps CW (s)": round(end_cw - start_cw, 6)
                }]
                
                df = pd.DataFrame(results)
                st.write("### Résultats de Commentz-Walter")
                st.dataframe(df)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=["Commentz-Walter"], y=[comparisons_cw], ax=ax)
                ax.set_title("Nombre de Comparaisons - Commentz-Walter")
                st.pyplot(fig)
            else:
                st.warning("Veuillez entrer un texte et des motifs.")
