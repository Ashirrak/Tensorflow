import streamlit as st
import time  
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import graphviz
import seaborn as sns
import altair as alt
import random
import string
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
    false_positives = {p: {} for p in patterns}  # Store positions & false matches

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
                    false_positives[pattern][i] = substring  # Store false match

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

    def visualize_failure_function(self):
        dot = graphviz.Digraph()

        if not hasattr(self, "nodes") or not isinstance(self.nodes, list):
            raise AttributeError("Error: 'nodes' should be a list. Check your Aho-Corasick class.")

        # Add nodes
        for i, node in enumerate(self.nodes):
            dot.node(str(i), label=f"State {i}")

        # Add failure links
        for i, node in enumerate(self.nodes):
            if hasattr(node, "fail") and node.fail is not None:
                dot.edge(str(i), str(node.fail), label="fail", color="red", style="dashed")

        return dot.source  # Return Graphviz source

    def visualize_output_function(self):
        dot = graphviz.Digraph()

        if not hasattr(self, "nodes") or not isinstance(self.nodes, list):
            raise AttributeError("Error: 'nodes' should be a list. Check your Aho-Corasick class.")

        # Add nodes
        for i, node in enumerate(self.nodes):
            label = f"State {i}"
            if hasattr(node, "output") and node.output:
                label += f"\nOutput: {', '.join(node.output)}"  # Show found patterns
            
            dot.node(str(i), label=label, shape="ellipse", color="blue")

        return dot.source  # Return Graphviz source

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
    "Aho-Corasick Multiple Pattern Matching",
    "Commentz-Walter",
    "Test",
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
            
            if occurrences:
                st.write("**Positions des occurrences:**", occurrences)
            else:
                st.write("**Aucune occurrence trouvée.**")
        
        elif algorithm == "Boyer-Moore":
            bad_char, comparisons, occurrences = boyer_moore_search(text, pattern)
            st.write("**Tableau dictionnaire:**", bad_char)
            st.write("**Nombre de comparaisons:**", comparisons)
        
            if occurrences:
                st.write("**Positions des occurrences:**", occurrences)
            else:
                st.write("**Aucune occurrence du motif n'a été trouvée dans le texte.**")

        elif algorithm == "Rabin-Karp (1 fonction de hachage)":
            start = time.time()
            results, comparisons = rabin_karp_multiple_search(text, patterns)
            end = time.time()
        
            st.write("**Résultats de Rabin-Karp (1 fonction de hachage) :**")
            st.write(f"**Nombre de Comparaisons :** {comparisons}")
            st.write(f"**Temps d'exécution :** {end - start:.6f} secondes")
        
            if any(results.values()):  # Check if there is at least one occurrence
                for pattern, occurrences in results.items():
                    st.write(f"Motif `{pattern}` trouvé aux positions : {occurrences if occurrences else 'Aucune occurrence trouvée'}")
            else:
                st.write("**Aucun motif n'a été trouvé dans le texte.**")

        elif algorithm == "Rabin-Karp (3 fonctions de hachage)":
            start = time.time()
            results, comparisons, false_positives = rabin_karp_bloom_filter(text, patterns, hash_functions)
            end = time.time()
            
            st.write("**Rabin-Karp avec 3 fonctions de hachage (Filtre de Bloom) :**")
            st.write(f"**Total Comparaisons:** {comparisons}")
            st.write(f"**Temps d'exécution:** {end - start:.6f} secondes")
        
            found_any = False  # Flag to check if any motif was found
        
            # Display found motifs
            for pattern, occurrences in results.items():
                if occurrences:
                    st.write(f"Motif `{pattern}` trouvé aux positions : {occurrences}")
                    found_any = True
                else:
                    st.write(f"Motif `{pattern}` non trouvé.")
        
            if not found_any:
                st.write("**Aucun motif trouvé dans le texte.**")
        
            # Display false positives
            has_false_positives = any(fp for fp in false_positives.values())
            if has_false_positives:
                st.write("### Faux Positifs Détectés:")
                false_pos_list = []
        
                for pattern, fp_dict in false_positives.items():
                    for pos, false_motif in fp_dict.items():
                        false_pos_list.append({"Motif Original": pattern, "Faux Motif": false_motif, "Position": pos})
        
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
        
                # Identify found and not found motifs
                found_patterns = set(pattern for _, pattern in results) if results else set()
                not_found_patterns = set(patterns) - found_patterns
        
                # Display results
                st.subheader("Results")
                if results:
                    results_df = pd.DataFrame(results, columns=["Position", "Pattern"])
                    st.dataframe(results_df)
                else:
                    st.write("**No occurrences found for any pattern.**")
        
                if not_found_patterns:
                    st.write("**Motifs not found:**", ", ".join(not_found_patterns))
        
                st.write(f"**Total comparisons:** {comparisons}")
                st.write(f"**Execution time:** {end_time - start_time:.6f} seconds")
        
                # Display Trie Automaton
                st.subheader("Automaton of Prefixes (Trie Structure)")
                st.graphviz_chart(ac.visualize_trie())
        
                # Display Search Path Automaton
                st.subheader("Search Path Automaton (Pattern Matching Process)")
                st.graphviz_chart(ac.visualize_search_path(text))
                st.subheader("Failure Function (Suppléance)")
                try:
                    st.graphviz_chart(ac.visualize_failure_function())
                except AttributeError as e:
                    st.error(f"Error: {e}. Your Aho-Corasick class may need adjustments.")
                
                # Display Output Function (Fonction de Sortie)
                st.subheader("Output Function (Fonction de Sortie)")
                st.graphviz_chart(ac.visualize_output_function())
        
        elif algorithm == "Commentz-Walter":
            if not patterns:
                st.warning("Please enter at least one pattern.")
            else:
                # Instantiate the Commentz-Walter class
                cw = CommentzWalter(patterns)  
                start = time.time()
                results, comparisons = cw.search(text)  
                end = time.time()
        
                st.write("**Commentz-Walter Algorithm:**")
                st.write(f"**Total Comparisons:** {comparisons}")
                st.write(f"**Execution Time:** {end - start:.6f} seconds")
        
                found_any = False  # Flag to check if any motif was found
        
                # Display found patterns
                for pattern, occurrences in results.items():
                    if occurrences:
                        st.write(f"Motif `{pattern}` trouvé aux positions : {occurrences}")
                        found_any = True
                    else:
                        st.write(f"Motif `{pattern}` non trouvé.")
        
                if not found_any:
                    st.write("**Aucun motif trouvé dans le texte.**")



if algorithm == "Test":
    
    test_choice = st.sidebar.radio(
        "Choix de tests",
        ["Test Morris-Pratt et Boyer-Moore", "Test Rabin-Karp (1 vs 3 fonctions de hachage)","Test Aho-Corasick","Test Commentz-Walter","Test AC vs CW"]
    )
    if test_choice == "Test Morris-Pratt et Boyer-Moore":
   # ===================== Streamlit App =====================
        st.subheader("Test Morris-Pratt et Boyer-Moore")
        
        # Inputs
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
        
                # Display table
                st.write("### Comparaison Morris-Pratt vs Boyer-Moore")
                st.dataframe(df)
        
                # ===  Graphs ===
                # Execution Time Comparison
                fig, ax = plt.subplots()
                ax.bar(["Morris-Pratt", "Boyer-Moore"], [df["Temps MP (s)"][0], df["Temps BM (s)"][0]], color=['blue', 'red'])
                ax.set_ylabel("Temps (s)")
                ax.set_title("Comparaison du Temps d'Exécution")
                st.pyplot(fig)
        
                # Number of Comparisons
                chart_data = pd.DataFrame({
                    "Algorithme": ["Morris-Pratt", "Boyer-Moore"],
                    "Comparaisons": [df["Comparaisons MP"][0], df["Comparaisons BM"][0]]
                })
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("Algorithme", sort=None),
                    y="Comparaisons",
                    color="Algorithme"
                )
                st.altair_chart(chart, use_container_width=True)
        
                # Best and Worst Case Analysis
                if comparisons_mp < comparisons_bm:
                    st.success(" **Morris-Pratt est plus efficace dans ce cas** (moins de comparaisons).")
                elif comparisons_mp > comparisons_bm:
                    st.warning(" **Boyer-Moore est plus efficace dans ce cas** (moins de comparaisons).")
                else:
                    st.info(" **Les deux algorithmes donnent des résultats similaires.**")
        
            else:
                st.warning("Veuillez entrer un texte et un motif.")
        
        # Importer tes fonctions d'algorithmes ici
        # from algorithms import morris_pratt_search, boyer_moore_search
        
        st.subheader("Test Morris-Pratt et Boyer-Moore avec différentes tailles de texte et motif")
        
        # Générer des textes aléatoires de différentes tailles
        text_sizes = [100, 500, 1000, 5000, 10000]  # Différentes tailles de texte
        pattern_sizes = [5, 10, 20, 50]  # Différentes tailles de motifs
        
        results = []
        
        for text_len in text_sizes:
            text_mp_bm = ''.join(random.choices(string.ascii_lowercase, k=text_len))  # Texte aléatoire
            
            for pattern_len in pattern_sizes:
                if pattern_len >= text_len:
                    continue  # Éviter les motifs plus grands que le texte
                
                pattern_mp_bm = text_mp_bm[:pattern_len]  # Extraire un motif du texte
        
                # Exécution de Morris-Pratt
                start_mp = time.time()
                _, comparisons_mp, _ = morris_pratt_search(text_mp_bm, pattern_mp_bm)
                end_mp = time.time()
        
                # Exécution de Boyer-Moore
                start_bm = time.time()
                _, comparisons_bm, _ = boyer_moore_search(text_mp_bm, pattern_mp_bm)
                end_bm = time.time()
        
                # Stocker les résultats
                results.append({
                    "Taille Texte": text_len,
                    "Taille Motif": pattern_len,
                    "Comparaisons MP": comparisons_mp,
                    "Temps MP (s)": round(end_mp - start_mp, 6),
                    "Comparaisons BM": comparisons_bm,
                    "Temps BM (s)": round(end_bm - start_bm, 6),
                })
        
        # Convertir en DataFrame
        df = pd.DataFrame(results)
        
        # Afficher les résultats
        st.write("### Comparaison Morris-Pratt vs Boyer-Moore")
        st.dataframe(df)
        
        # Tracer les courbes d'évolution des temps d'exécution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Taille Texte"], df["Temps MP (s)"], marker='o', label="Morris-Pratt", color="blue")
        ax.plot(df["Taille Texte"], df["Temps BM (s)"], marker='s', label="Boyer-Moore", color="red")
        ax.set_xlabel("Taille du Texte")
        ax.set_ylabel("Temps d'exécution (s)")
        ax.set_title("Évolution du Temps d'Exécution en fonction de la Taille du Texte")
        ax.legend()
        st.pyplot(fig)
        
        # Tracer les courbes du nombre de comparaisons
        chart = alt.Chart(df).mark_line(point=True).encode(
            x="Taille Texte",
            y="Comparaisons MP",
            color=alt.value("blue")
        ).properties(title="Comparaisons Morris-Pratt vs Boyer-Moore").interactive()
        
        chart += alt.Chart(df).mark_line(point=True).encode(
            x="Taille Texte",
            y="Comparaisons BM",
            color=alt.value("red")
        )
        
        st.altair_chart(chart, use_container_width=True)

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
    
                # Compute total false positive count
                false_positive_count = sum(len(fp_dict) for fp_dict in false_positives.values())
    
                # Create main results dataframe
                results = [
                    {"Méthode": "RK1", "Comparaisons": comparisons_rk1, "Temps (s)": round(end_rk1 - start_rk1, 6), "Faux Positifs": 0},
                    {"Méthode": "RK3", "Comparaisons": comparisons_rk3, "Temps (s)": round(end_rk3 - start_rk3, 6), "Faux Positifs": false_positive_count}
                ]
    
                df = pd.DataFrame(results)
                st.write("### Résultats de la comparaison")
                st.dataframe(df)
    
                # Display motifs found and positions
                motifs_results = []
                for method, result in zip(["RK1", "RK3"], [results_rk1, results_rk3]):
                    motifs_results.append({
                        "Méthode": method,
                        "Positions trouvées": result.get(pattern_rk, "Aucun"),
                        "Nombre de motifs trouvés": len(result.get(pattern_rk, []))
                    })
    
                df_motifs = pd.DataFrame(motifs_results)
                st.write("### Motifs trouvés et leurs positions")
                st.dataframe(df_motifs)
    
                # Display false positive motifs if any exist
                if false_positive_count > 0:
                    false_pos_list = []
                    for pos, false_motif in false_positives.get(pattern_rk, {}).items():
                        false_pos_list.append({
                            "Motif Original": pattern_rk,
                            "Faux Motif": false_motif,
                            "Position": pos
                        })
    
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
        
        
        # Importer tes fonctions d'algorithmes ici
        # from algorithms import rabin_karp_multiple_search, rabin_karp_bloom_filter
        
        st.subheader("Comparaison des performances entre RK1 et RK3 avec différentes tailles de texte et motif")
        
        # Générer des textes aléatoires de différentes tailles
        text_sizes = [100, 500, 1000, 5000, 10000]  # Différentes tailles de texte
        pattern_sizes = [5, 10, 20, 50]  # Différentes tailles de motifs
        num_filters = st.slider("Nombre de filtres Bloom pour RK3", 1, 5, 3,key="slider_filters")
        
        results = []
        
        for text_len in text_sizes:
            text_rk = ''.join(random.choices(string.ascii_lowercase, k=text_len))  # Texte aléatoire
            
            for pattern_len in pattern_sizes:
                if pattern_len >= text_len:
                    continue  # Éviter les motifs plus grands que le texte
                
                pattern_rk = text_rk[:pattern_len]  # Extraire un motif du texte
                hash_functions_set = [(50 + i, 31 + i * 2) for i in range(num_filters)]
        
                # Exécution de RK1
                start_rk1 = time.time()
                results_rk1, comparisons_rk1 = rabin_karp_multiple_search(text_rk, [pattern_rk])
                end_rk1 = time.time()
        
                # Exécution de RK3
                start_rk3 = time.time()
                results_rk3, comparisons_rk3, false_positives = rabin_karp_bloom_filter(text_rk, [pattern_rk], hash_functions_set)
                end_rk3 = time.time()
        
                # Nombre total de faux positifs
                false_positive_count = sum(len(fp_dict) for fp_dict in false_positives.values())
        
                # Stocker les résultats
                results.append({
                    "Taille Texte": text_len,
                    "Taille Motif": pattern_len,
                    "Comparaisons RK1": comparisons_rk1,
                    "Temps RK1 (s)": round(end_rk1 - start_rk1, 6),
                    "Comparaisons RK3": comparisons_rk3,
                    "Temps RK3 (s)": round(end_rk3 - start_rk3, 6),
                    "Faux Positifs RK3": false_positive_count
                })
        
        # Convertir en DataFrame
        df = pd.DataFrame(results)
        
        # Afficher les résultats
        st.write("### Résultats de la comparaison entre RK1 et RK3")
        st.dataframe(df)
        
        # Tracer les courbes d'évolution des temps d'exécution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Taille Texte"], df["Temps RK1 (s)"], marker='o', label="RK1", color="blue")
        ax.plot(df["Taille Texte"], df["Temps RK3 (s)"], marker='s', label="RK3", color="red")
        ax.set_xlabel("Taille du Texte")
        ax.set_ylabel("Temps d'exécution (s)")
        ax.set_title("Évolution du Temps d'Exécution en fonction de la Taille du Texte")
        ax.legend()
        st.pyplot(fig)
        
        # Tracer les courbes du nombre de comparaisons
        chart = alt.Chart(df).mark_line(point=True).encode(
            x="Taille Texte",
            y="Comparaisons RK1",
            color=alt.value("blue")
        ).properties(title="Comparaisons RK1 vs RK3").interactive()
        
        chart += alt.Chart(df).mark_line(point=True).encode(
            x="Taille Texte",
            y="Comparaisons RK3",
            color=alt.value("red")
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Graphique des faux positifs
        fig_fp, ax_fp = plt.subplots(figsize=(6, 4))
        ax_fp.bar(["RK1", "RK3"], [0, df["Faux Positifs RK3"].sum()], color=['blue', 'red'])
        ax_fp.set_title("Nombre de Faux Positifs")
        st.pyplot(fig_fp)

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
            st.subheader("Test et analyse de l’algorithme Aho-Corasick avec différentes tailles de texte et motifs")
            
            # Define text and pattern sizes
            text_sizes = [100, 500, 1000, 5000, 10000]
            pattern_counts = [2, 5, 10, 20]
            
            results = []
            
            for text_len in text_sizes:
                text_ac = ''.join(random.choices(string.ascii_lowercase, k=text_len))  # Generate random text
                
                for num_patterns in pattern_counts:
                    patterns_ac = [text_ac[i: i + 5] for i in range(0, min(num_patterns * 5, text_len - 5), 5)]
                    
                    # Initialize Aho-Corasick Automaton
                    ac = AhoCorasick()
                    for pattern in patterns_ac:
                        ac.insert(pattern, len(pattern))
                    ac.build()
            
                    # Measure execution time
                    start_ac = time.time()
                    results_ac, comparisons_ac = ac.search(text_ac)
                    end_ac = time.time()
            
                    # Store results
                    results.append({
                        "Taille du Texte": text_len,
                        "Nombre de Motifs": num_patterns,
                        "Comparaisons AC": comparisons_ac,
                        "Temps AC (s)": round(end_ac - start_ac, 6)
                    })
            
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            st.write("### Résultats de Aho-Corasick")
            st.dataframe(df)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.lineplot(data=df, x="Taille du Texte", y="Temps AC (s)", marker="o", label="Temps AC")
            ax.set_title("Temps d'exécution en fonction de la taille du texte")
            st.pyplot(fig)
            
            fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
            sns.lineplot(data=df, x="Taille du Texte", y="Comparaisons AC", marker="s", label="Comparaisons AC")
            ax_comp.set_title("Nombre de Comparaisons en fonction de la taille du texte")
            st.pyplot(fig_comp)

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
        st.subheader("Test et analyse de l’algorithme Commentz-Walter")
        
        # Define different text sizes for testing
        text_sizes = [100, 500, 1000, 5000, 10000]  # Various text sizes
        pattern_sizes = [5, 10, 20, 50]  # Different pattern sizes
        
        results = []
        
        for text_len in text_sizes:
            text_cw = ''.join(random.choices(string.ascii_lowercase, k=text_len))  # Generate random text
            
            for pattern_len in pattern_sizes:
                if pattern_len >= text_len:
                    continue  # Skip if pattern is longer than text
                
                pattern_cw = text_cw[:pattern_len]  # Extract a pattern from text
                
                cw = CommentzWalter([pattern_cw])
                start_cw = time.time()
                results_cw, comparisons_cw = cw.search(text_cw)
                end_cw = time.time()
                
                # Store results
                results.append({
                    "Taille du Texte": text_len,
                    "Taille du Motif": pattern_len,
                    "Comparaisons CW": comparisons_cw,
                    "Temps CW (s)": round(end_cw - start_cw, 6)
                })
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Display results table
        st.write("### Résultats de Commentz-Walter")
        st.dataframe(df)
        
        # Visualization: Execution time evolution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Taille du Texte"], df["Temps CW (s)"], marker='o', label="Commentz-Walter", color="purple")
        ax.set_xlabel("Taille du Texte")
        ax.set_ylabel("Temps d'exécution (s)")
        ax.set_title("Évolution du Temps d'Exécution en fonction de la Taille du Texte")
        ax.legend()
        st.pyplot(fig)
        
        # Visualization: Number of comparisons
        fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
        sns.barplot(x=df["Taille du Texte"], y=df["Comparaisons CW"], ax=ax_comp, palette="viridis")
        ax_comp.set_xlabel("Taille du Texte")
        ax_comp.set_ylabel("Nombre de Comparaisons")
        ax_comp.set_title("Comparaisons de l'algorithme Commentz-Walter")
        st.pyplot(fig_comp)

    elif test_choice == "Test AC vs CW":     
        st.subheader("Comparaison des Algorithmes AC vs CW")
        
        text_sizes = [100, 500, 1000, 5000, 10000]  # Text sizes to test
        pattern_counts = [2, 5, 10, 20]  # Number of patterns to test
        
        results = []
        
        for text_len in text_sizes:
            text = ''.join(random.choices(string.ascii_lowercase, k=text_len))  # Generate random text
            
            for num_patterns in pattern_counts:
                patterns = [text[i: i + 5] for i in range(0, min(num_patterns * 5, text_len - 5), 5)]
                
                # Test Aho-Corasick
                ac = AhoCorasick()
                for pattern in patterns:
                    ac.insert(pattern, len(pattern))
                ac.build()
                start_ac = time.time()
                _, comparisons_ac = ac.search(text)
                end_ac = time.time()
                
                # Test Commentz-Walter
                cw = CommentzWalter(patterns)
                start_cw = time.time()
                _, comparisons_cw = cw.search(text)
                end_cw = time.time()
                
                results.append({
                    "Taille du Texte": text_len,
                    "Nombre de Motifs": num_patterns,
                    "Comparaisons AC": comparisons_ac,
                    "Temps AC (s)": round(end_ac - start_ac, 6),
                    "Comparaisons CW": comparisons_cw,
                    "Temps CW (s)": round(end_cw - start_cw, 6)
                })
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        st.write("### Résultats Comparatifs")
        st.dataframe(df)
        
        # Visualization: Execution time
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=df, x="Taille du Texte", y="Temps AC (s)", marker="o", label="Aho-Corasick")
        sns.lineplot(data=df, x="Taille du Texte", y="Temps CW (s)", marker="s", label="Commentz-Walter")
        ax.set_title("Comparaison du Temps d'Exécution")
        st.pyplot(fig)
        
        # Visualization: Number of comparisons
        fig_comp, ax_comp = plt.subplots(figsize=(8, 5))
        sns.lineplot(data=df, x="Taille du Texte", y="Comparaisons AC", marker="o", label="Aho-Corasick")
        sns.lineplot(data=df, x="Taille du Texte", y="Comparaisons CW", marker="s", label="Commentz-Walter")
        ax_comp.set_title("Comparaison du Nombre de Comparaisons")
        st.pyplot(fig_comp)
