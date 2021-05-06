
class PhraseMetadata(object):
    def __init__(self, op, acore, bcore, afull, bfull):
        self.acore, self.bcore = acore, bcore
        self.afull, self.bfull = afull, bfull
        #self.acore, self.bcore = araw, braw
        
        self.op = op
        self.edit_dist = len(self.acore) + len(self.bcore)
        self.sem_dist = -1
        self._set_dep_and_pos()
        self.tag = None

    def _set_dep_and_pos(self):
        if self.op == "insert" or (self.op != "delete" and (not _normalize_span(self.acore) or _normalize_span(self.acore).root.dep_ == "det")):
            core = self.bcore
        else:
            core = self.acore
        core = _normalize_span(core)
        self.dep, self.pos = core.root.dep_, core.root.pos_

    def short_str(self):
        return f"[{self.op}] {self.acore} -> {self.bcore}"

    def __repr__(self):
        string = self.short_str()
        #string += f"\n\t== {self.afull}"
        #string += f"\n\t== {self.bfull}"
        #string += f"\n\tstructure remain: {self._is_structure_remain()}"
        #string += f"\n\tlocal: {self._is_local()}"
        string += f"\n\ttag: {self.tag}"
        string += f"\n\tdep: {unify_tags(self.dep)}"
        string += f"\n\tsem_dist: {self.sem_dist:.3f}"
        return string
    
    def _is_local(self):
        # TODO: change this part maybe?
        # How many different dep trees are changed
        a, b = (self.acore, self.bcore) if self.op == "replace" else (self.afull, self.bfull)
        return max(len(self.acore), len(self.bcore)) <= 5 and \
            (len(get_top_tokens(a)) == 1 and len(get_top_tokens(b)) == 1)

    def _is_structure_remain(self):
        t1, t2 = (self.acore, self.bcore) if self.op == "replace" else (self.afull,self.bfull)
        return is_same_structure(t1, t2)

    def compute_sem_dist(self, sentence_similarity, reset=False):
        #print(self.afull, self.bfull, sentence_similarity(self.afull.text, self.bfull.text))
        #print(self.acore, self.bcore, sentence_similarity(self.acore.text, self.bcore.text))
        if reset:
            self.sem_dist = -1
        if self.sem_dist == -1: 
            l1 = set([a.lemma_.lower() for a in self.acore])
            l2 = set([a.lemma_.lower() for a in self.bcore])
            stopset = set(["a", "an", "the", "this", "that", "be"])
            if (l1-l2).issubset(stopset) and (l2-l1).issubset(stopset):
                self.sem_dist = 0.01
            else:
                self.sem_dist = min(
                    sentence_similarity(self.afull.text, self.bfull.text),
                    sentence_similarity(self.acore.text, self.bcore.text),
                ) if sentence_similarity else -1

    def compute_metadata(self, sentence_similarity, is_overwrite_local=False):
        tag = None
        is_local, is_structure_remain = self._is_local(), self._is_structure_remain()
        #print("structure_remain", is_structure_remain)
        acore, bcore = _normalize_span(self.acore), _normalize_span(self.bcore)
        afull, bfull = _normalize_span(self.afull), _normalize_span(self.bfull)
        aroot = acore.root if acore else None
        broot = bcore.root if bcore else None

        if is_overwrite_local:
            is_local = True

        change_set = [aroot.text.lower() if aroot else "", broot.text.lower() if broot else ""]
        def is_wordnet_change(aroot, broot):
            return not (aroot.is_stop and broot.is_stop) and aroot.lemma_ not in ["have", "be", "get"] and broot.lemma_ not in ["have", "be"]
        xor = lambda func: (func(acore) and not func(bcore)) or (func(bcore) and not func(acore))
        ### negation
        negs1, neg_heads1 = get_negations(acore)
        negs2, neg_heads2 = get_negations(bcore)
        #print(negs1, negs2, neg_heads1, neg_heads2)
        if is_local and tag is None and (
            (negs1 != negs2) or \
            (self.op == "replace" and change_set == set(["and", "but"])) or \
            xor(is_subjunctive)):
            tag = "negation"
        ### quantifiers
        quants1, nums1, base1 = get_quantifiers(acore)
        quants2, nums2, base2 = get_quantifiers(bcore)
        #print(base1, base2)
        #print("get_quantifiers", quants1, nums1, base1)
        #print("get_quantifiers", quants2, nums2, base2)
        #print(quants1, quants2, nums1, nums2)
        # TODO: what is this one doing?
        if tag is None and is_local and (base1 == base2 or \
            self.op != "replace"  or \
            (self.op == "replace" and acore.root.head.lemma_ == bcore.root.head.lemma_)):
            if quants1 and quants2 and quants1 != quants2: #quants1 and quants2 and
                tag = "quantifier"
            n1, n2 = list(nums1-nums2)[0] if nums1-nums2 else DEFAULT_NUM, list(nums2-nums1)[0] if nums2-nums1 else DEFAULT_NUM
            if tag is None and ((n1 != DEFAULT_NUM) or (n2 != DEFAULT_NUM)):
                tag = "quantifier"
            if set([acore.text.lower(),bcore.text.lower()]).issubset(set(["a", "one", "an"])):
                tag = None
        ### split the nouns
        # TODO: should this be just for NOUNs

        def get_span_info(func, span, full_span): 
            return func(span) if len(span) and func(span) else func(full_span)
        if tag is None:
            p1 = get_span_info(get_passive_token, acore, afull)
            p2 = get_span_info(get_passive_token, bcore, bfull)
            #print(is_subjunctive(self.acore), is_subjunctive(self.bcore))
            tense1, v1 = get_span_info(get_tense, acore, afull)
            tense2, v2 = get_span_info(get_tense, bcore, bfull)
            if (xor(is_passive) and p1 and p1 == p2) or (tense1 != tense2 and v1 == v2):
                tag = "restructure"
        if tag is None and self.op == "replace" and is_local and is_structure_remain:
            hypers1, hypers2 = get_hypernyms(aroot), get_hypernyms(broot)
            #print(aroot, hypers1)
            #print(broot, hypers2)
            if is_wordnet_change(aroot, broot) and (acore.lemma_.lower() in hypers2 or acore.lower_ in hypers2):
                tag = "lexical"
            elif is_wordnet_change(aroot, broot) and (bcore.lemma_.lower() in hypers1 or bcore.lower_ in hypers1):
                tag = "lexical"
            else:
                syms = get_wordnet_info(all_possible_synonyms, acore)
                ayms = get_wordnet_info(all_possible_antonyms, acore)
                #if False and bcore.lemma_ in syms or bcore.text.lower() in syms: 
                #    tag = "paraphrase"
                #el
                if bcore.lemma_ in ayms or bcore.text.lower() in ayms:
                    tag = "lexical"
                else:
                    #self.compute_sem_dist(sentence_similarity)
                    #if False and self.sem_dist <= 0.1 and self.sem_dist >= 0:
                    #    tag = "paraphrase" 
                    #el
                    if is_single_change(acore) and is_single_change(bcore):
                        tag = "lexical"
                    else:
                        tag = "resemantic"

        if tag is None and self.op != "replace" and is_structure_remain:
            #self.compute_sem_dist(sentence_similarity)
            #if self.sem_dist >= 0 and self.sem_dist <= 0.1: tag = "paraphrase"
            #else: tag = "constraint"
            #self.compute_sem_dist(sentence_similarity)
            #tag = "paraphrase" if self.sem_dist <= 0.1 and self.sem_dist >= 0 else self.op
            tag = self.op # TODO: should we keep the semantic tag?
        if tag is None and is_local and not is_structure_remain:
            #self.compute_sem_dist(sentence_similarity)
            #tag = "paraphrase" if self.sem_dist <= 0.1 and self.sem_dist >= 0 else "restructure"
            tag = "other"
        if tag is None and not is_local: tag = "other"
        if tag is None: tag = "global"
        #print(tag, self.tag)
        self.tag = tag
        

class SentenceMetadata(object):
    def __init__(self, ops):
        self.phrases = [
            PhraseMetadata(op.op, 
                op.fromz_core, op.toz_core, 
                op.fromz_full, op.toz_full,
            ) for op in ops]
        self.primary = None
    def to_dict(self):
        return self.primary.to_dict() if self.primary else {}
    def export_sentence(self):
        if self.primary:
            return {
                "atext": self.primary.acore.doc.text,
                "btext":self.primary.bcore.doc.text
            }
        else:
            return {"atext": "", "btext": ""}
    def __repr__(self):
        if self.primary:
            primary = str(self.primary)
            others = [str(p) for p in self.phrases if str(p) != primary]
            strs = [primary] + others
        else:
            strs = [str(p) for p in self.phrases]
        string = "\n".join(strs)
        return string

    def short_str(self):
        string = "\n".join([p.short_str() for p in self.phrases])
        return string

    def _get_change_intersections(self):
        exclude_list = set(["be", "this", "there", "the", "a", "an", "that", "those", "-pron-"])
        pos_list = ["NOUN", "ADJ", "ADV","NUM", "PRON", "PROPN", "VERB"]
        achanges, bchanges = set(), set()
        #print(self.nonoverlap_phrases(use_full=False))
        phrases = self.nonoverlap_phrases(use_full=False)
        filter_word = lambda s: not s.is_punct and s.pos_ in pos_list and s.lemma_.lower() not in exclude_list
        for p in phrases: #self.phrases:
            cur_a = set([s.lemma_.lower() for s in p.acore if filter_word(s)])
            cur_b = set([s.lemma_.lower() for s in p.bcore if filter_word(s)])
            achanges.update(cur_a-cur_b)
            bchanges.update(cur_b-cur_a)
        intersect = achanges.intersection(bchanges)
        return achanges, bchanges, intersect, phrases
        
    def get_major_phrases(self):
        if not self.primary:
            return []
        primary = self.primary
        allowed_phrase_combinations = []
        if primary.tag in ["shuffle"]:
            new_phrases = []
            achanges, bchanges, intersect, phrases = self._get_change_intersections()
            for p in phrases:
                if any([i in p.acore.lemma_.lower() for i in intersect]):
                    bcore = p.bcore.doc[p.bcore.start:p.bcore.start]
                    np = PhraseMetadata("delete", p.acore, bcore, p.afull, p.bfull, "", "")
                    np.tag = "shuffle"
                    new_phrases.append(np)
                if any([i in p.bcore.lemma_.lower() for i in intersect]):
                    acore = p.acore.doc[p.acore.start:p.acore.start]
                    np = PhraseMetadata("insert", acore, p.bcore, p.afull, p.bfull, "", "")
                    np.tag = "shuffle"
                    new_phrases.append(np)
            allowed_phrase_combinations.append(new_phrases)
        elif primary.tag == "restructure":
            pass
        elif primary.tag in ["lexical", "resemantic"]:
            allowed_phrase_combinations.append([primary])
        else:
            phrases = self.nonoverlap_phrases(use_full=False)
            if len(phrases) > 1 or primary.tag not in ["lexical", "resemantic"]:
                allowed_phrase_combinations.append(phrases)
        return allowed_phrase_combinations

    def compute_metadata(self, sentence_similarity, is_select_primary=True):
        for p in self.phrases:
            p.compute_metadata(sentence_similarity)
            #print(p)
        if not is_select_primary or not self.phrases:
            return
        self.select_primary(sentence_similarity)
        
        #print("--- primary ---")
        #print(self.primary)
    
    def select_primary(self, sentence_similarity):
        def get_order(p, treat_insert_diff=False):
            order = {
                "global": -1,
                #"lexical":3,
                #"hyponym":3,
                #"wordnet":3,
                #"other": 3,
                #"quantifier": 0,
                #"constraint": 1,
                #"restructure": 2,
                "paraphrase": 1000,
                "restructure": 1,
            }
            if treat_insert_diff:
                order["insert"] = 500
                order["delete"] = 500
            return order[p.tag] if p.tag in order else 2
        if len(self.phrases) == 1: 
            self.primary = self.phrases[0]
            return
        for p in self.phrases:
            p.compute_sem_dist(sentence_similarity)
        
        achanges, bchanges, intersect, _ = self._get_change_intersections()
        #print(intersect, achanges, bchanges)
        # get phrases if they relate to the intersection
        has_shuffle = len(intersect) > 0 and (len(intersect) == len(achanges) or len(intersect) == len(bchanges)) \
            and "there" not in achanges and \
            "there" not in bchanges
        # if re-structuring, compare the depth of all the dependency trees and the children relationship
        # if some other core is within the restructure full, then prioritize the restructruing
        # rationale: they would be related to the restructuring.
        phrases_for_sort = []
        p_restructures = self.nonoverlap_phrases(use_full=False, phrases=[p for p in self.phrases if p.tag == "restructure"])

        for p in self.phrases:
            if p.tag == "restructure":
                continue
            if any([is_child_phrase(p.acore, pr.afull) and is_child_phrase(p.bcore, pr.bfull) for pr in p_restructures]):
                continue
            phrases_for_sort.append(p)
        #if phrases_for_sort:
        phrases_for_sort += p_restructures
        #else:
        #    phrases_for_sort = self.phrases
        phrases_for_sort = sorted(phrases_for_sort, key=lambda p: (get_order(p), -p.sem_dist))
        #print([(p, get_order(p)) for p in phrases_for_sort])
        #phrases_for_shuffle = sorted(phrases_for_sort, key=lambda p: (-p.sem_dist))
        #print(phrases_for_sort)
        if len(phrases_for_sort) == 0: 
            self.primary = None
            return
        #if len(intersect) / len(achanges) >= 0.5 and len(intersect) / len(bchanges) >= 0.5:
        #    print("shuffle")
        has_multiple = len(set([p.op for p in phrases_for_sort if p.op in ["insert", "delete"] and \
            p.tag in ["insert", "delete", "restructure"] ])) > 1
        #print(phrases_for_sort[0].op in ["insert", "delete"], has_shuffle, has_multiple)

        cur_a = set([s.lemma_.lower() for s in phrases_for_sort[0].acore if not s.is_stop and not s.is_punct])
        cur_b = set([s.lemma_.lower() for s in phrases_for_sort[0].bcore if not s.is_stop and not s.is_punct])

        if any([p.tag == "global" for p in phrases_for_sort]) or \
            len([p for p in phrases_for_sort if p.tag != "paraphrase"]) >= 5:
            self.primary = phrases_for_sort[0]
            self.primary.tag = "global"
        elif has_shuffle: #and intersect.intersection(cur_a.union(cur_b)) and all([p.tag != "restructure" for p in self.phrases]):
            #phrases_shuffle = [s for s in self.phrases if any([s.lemma_ in intersect for s in list(p.acore) + list(p.bcore)])]
            self.primary = phrases_for_sort[0]
            if all([p.tag != "restructure" for p in self.phrases]): 
                self.primary.tag = "shuffle"
        elif phrases_for_sort[0].op in ["insert", "delete"] and has_multiple:
            self.primary = phrases_for_sort[0]
            inserts = [p for p in phrases_for_sort if p.op == "insert"]
            deletes = [p for p in phrases_for_sort if p.op == "delete"]
            self.primary.tag = "resemantic"
            for i, d in itertools.product(inserts, deletes):
                if not is_same_structure(i.bcore, d.acore):
                    self.primary.tag = "global"
                    break
        else:
            self.primary = phrases_for_sort[0]
        if self.primary.tag == "other": self.primary.tag = "global"

    def nonoverlap_phrases(self, use_full, merge_entire=False, phrases=None):
        if not phrases:
            phrases = self.phrases
        a, b = ("afull", "bfull") if use_full else ("acore", "bcore")
        if merge_entire:

            acore_starts = [p.acore.start for p in phrases]
            acore_ends = [p.acore.end for p in phrases]
            bcore_starts = [p.bcore.start for p in phrases]
            bcore_ends = [p.bcore.end for p in phrases]

            afull_starts = [p.afull.start for p in phrases]
            afull_ends = [p.afull.end for p in phrases]
            bfull_starts = [p.bfull.start for p in phrases]
            bfull_ends = [p.bfull.end for p in phrases]

            adoc, bdoc = phrases[0].acore.doc, phrases[0].bcore.doc
            acore = adoc[min(acore_starts):max(acore_ends)]
            bcore = bdoc[min(bcore_starts):max(bcore_ends)]
            afull = adoc[min(afull_starts):max(afull_ends)]
            bfull = bdoc[min(bfull_starts):max(bfull_ends)]
            if not acore: op = "insert"
            elif not bcore: op = "delete"
            else: op = "replace"
            """
            print(pstart.acore)
            print(pstart.bcore)
            print(afull.acore)
            print(pstart.bcore)
            print(op, acore, bcore, afull, bfull, acore, bcore)
            """
            try:
                normalized_phrases = [PhraseMetadata(op, acore, bcore, afull, bfull, acore, bcore)]
            except:
                normalized_phrases = []
        else:
            phrases = sorted(phrases, key=lambda l: (len(getattr(l, a)) , len(getattr(l, b)) ))
            normalized_phrases = []
            for idx, n in enumerate(phrases):
                is_subset = False
                for m in phrases[idx+1:]:
                    #print(n.short_str(), getattr(n, a).start, getattr(n, a).end, getattr(n, b).start, getattr(n, b).end)
                    #print(m.short_str(), getattr(m, a).start, getattr(m, a).end, getattr(m, b).start, getattr(m, b).end)
                    #if (getattr(n, a).start >= getattr(m, a).start and getattr(n, a).end <= getattr(m, a).end or n.op == "insert") and \
                    #    (getattr(n, b).start >= getattr(m, b).start and getattr(n, b).end <= getattr(m, b).end  or n.op == "delete"):
                    #print(getattr(n, a), getattr(m, a))
                    #print(getattr(n, b), getattr(m, b))
                    if is_subphrase(getattr(n, a), getattr(m, a)) and is_subphrase(getattr(n, b), getattr(m, b)):
                        is_subset = True
                        break
                    elif not (getattr(n, a).start > getattr(m, a).end or getattr(m, a).start > getattr(n, a).end):
                        return []
                if not is_subset:
                    normalized_phrases.append(n)
            normalized_phrases = sorted(normalized_phrases, key=lambda p: p.afull.start)
        return normalized_phrases