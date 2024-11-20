import numpy as np
import faiss
from pynndescent import NNDescent
import time

class GeneralizedIndex:
    def __init__(
        self,
        data,
        n_neighbors=15,
        metric='euclidean',
        metric_kwds=None, 
        random_state=None, # n_trees, n_iters, max_candidates
        low_memory=False,
        n_jobs=-1,
        verbose=False, # compressed=False
        use_pynndescent=True,
        faiss_index_factory_str=None,
        faiss_kwds=None
        ):
        self.use_pynndescent = use_pynndescent
        self.faiss_index_factory_str = faiss_index_factory_str
        self.faiss_kwds:dict = faiss_kwds if faiss_kwds is not None and isinstance(faiss_kwds,dict) else None
        self._raw_data = data
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.random_state = random_state
        self.verbose = verbose
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        if self.n_jobs == -1:
            omp_num_threads = faiss.omp_get_max_threads()
        else:
            omp_num_threads = n_jobs
        if self.verbose:
            print(f'Setting {omp_num_threads} threads for Index.')
        faiss.omp_set_num_threads(num_threads=omp_num_threads)
        if self.metric in ['euclidean','l2','L2']:
            faiss_metric = faiss.METRIC_L2
        elif self.metric == 'cosine' or self.metric.lower() in ['inner','ip']:
            faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric == 'canberra':
            faiss_metric = faiss.METRIC_Canberra
        elif not self.use_pynndescent:
            if self.verbose:
                print(f"Metric {self.metric} not accepted by Faiss. Using PyNNDescent.")
            self.use_pynndescent=True
        if self.use_pynndescent:
            start = time.time()
            n_trees = min(64, 5 + int(round((self._raw_data.shape[0]) ** 0.5 / 20.0)))
            n_iters = max(5, int(round(np.log2(self._raw_data.shape[0]))))
            self.search_index = NNDescent(
                data=self._raw_data,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                metric_kwds=self.metric_kwds,
                random_state=self.random_state,
                n_trees = n_trees,
                n_iters = n_iters,
                max_candidates=60,
                low_memory=self.low_memory,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                compressed=False
            )
            index_creation_time = None
            knn_construction_time = None
            tot_time = time.time() - start
            self._angular_trees = self.search_index._angular_trees
            knn_indices, knn_dists = self.search_index.neighbor_graph
        else:
            start = time.time()
            self._angular_trees = False
            # Faiss Index Initialization: index_factory vs kwds parameters matching
            if self.faiss_index_factory_str is not None:
                assert isinstance(self.faiss_index_factory_str,str), "The provided index factory string is not a string"
                # If the cosine metric is used, the vectors have to be L2-normalized before adding them or querying them
                if self.metric.lower()=='cosine':
                    self._raw_data_norm = np.copy(self._raw_data).astype(np.float32)
                    faiss.normalize_L2(self._raw_data_norm)
                try:
                    self.search_index = faiss.index_factory(self._raw_data.shape[1], self.faiss_index_factory_str, faiss_metric)
                    self.search_index.verbose = self.verbose # Maybe check index type before assignment
                    if self.verbose:
                        print("Built the index from the factory string. Index Class:",type(self.search_index).__name__)
                except Exception as e:
                    print("Encountered an exception while initializing index with the provided factory string: ",e)
                    raise(e)
            else:
                self.search_index = faiss.IndexHNSWFlat(
                    self._raw_data.shape[1],
                    self.n_neighbors,
                    faiss_metric
                )
            # Initilize general index pointer to the core Index
            downcasted_index_classname = None
            if type(self.search_index).__name__ == 'IndexPreTransform':
                internal_index = self.search_index.index
                downcasted_index_pointer = faiss.downcast_index(internal_index)
                downcasted_index_classname = type(downcasted_index_pointer).__name__
            
            if downcasted_index_classname is not None:
                index_pointer = downcasted_index_pointer
            else:
                index_pointer = self.search_index
            # if type(self.search_index).__name__ == 'IndexHNSWFlat':
            if self.faiss_kwds is not None:
                if type(index_pointer).__name__ == 'IndexHNSWFlat':
                # in self.search_index.__class__.mro():
                    if self.faiss_kwds is not None and 'efConstruction' in self.faiss_kwds.keys():
                        # self.search_index.hnsw.efConstruction = faiss_kwds.get('efConstruction')
                        index_pointer.hnsw.efConstruction = faiss_kwds.get('efConstruction')
                    if self.faiss_kwds is not None and 'efSearch' in self.faiss_kwds.keys():
                        # self.search_index.hnsw.efSearch = faiss_kwds.get('efSearch')
                        index_pointer.hnsw.efSearch = faiss_kwds.get('efSearch')
                # TODO ...
                elif type(index_pointer).__name__.startswith('IndexIVF'):
                    if 'nlist' in self.faiss_kwds.keys():
                        index_pointer.nlist = faiss_kwds.get('nlist')
                    if 'nprobe' in self.faiss_kwds.keys():
                        index_pointer.nprobe = faiss_kwds.get('nprobe')
                elif type(index_pointer).__name__ == 'IndexIVFFlat':
                    ## TODO
                    if 'nlist' in self.faiss_kwds.keys():
                        index_pointer.nlist = faiss_kwds.get('nlist')
                    if 'nprobe' in self.faiss_kwds.keys():
                        index_pointer.nprobe = faiss_kwds.get('nprobe')
                elif type(index_pointer).__name__ == 'IndexIVFPQFastScan':
                    ## TODO
                    if 'nlist' in self.faiss_kwds.keys():
                        index_pointer.nlist = faiss_kwds.get('nlist')
                    if 'nprobe' in self.faiss_kwds.keys():
                        index_pointer.nprobe = faiss_kwds.get('nprobe')
            
            # Index Metric setting
            
                
            # Train the index/quantizer if present
            try:
                if not self.search_index.is_trained:
                    self.search_index.train(self._raw_data)
            except Exception as e:
                print("Encountered an error while training the Faiss index: ",e)
                raise(e)
            # Add the vectors to the index (costly operation)
            try:
                self.search_index.add(self._raw_data) if metric != 'cosine' else self.search_index.add(self._raw_data_norm)
            except Exception as e:
                print("Encountered an error while adding the training data to the Faiss index: ",e)
                raise(e)
            index_creation_time = time.time() - start
            try:
                if self.metric == 'cosine':
                    knn_dists, knn_indices = self.search_index.search(self._raw_data_norm, self.n_neighbors)
                else:
                    knn_dists, knn_indices = self.search_index.search(self._raw_data, self.n_neighbors)
            except Exception as e:
                print("Encountered an error while building the knn graph with the faiss index: ",e)
                raise(e)
            if self.metric in ['euclidean','l2','L2']:
                knn_dists = np.sqrt(knn_dists)
            elif self.metric.lower() in ['cosine','ip', 'inner']:
                knn_dists = 1-knn_dists
            tot_time = time.time()-start
            knn_construction_time = tot_time - index_creation_time
        self.neighbor_graph = (knn_indices,knn_dists)
        self._times = {
            'index_creation':index_creation_time,
            'knn_construction':knn_construction_time,
            'total':tot_time
        }
    
    def query(self,X,n_neighbors):
        epsilon = 0.24 if self._knn_search_index._angular_trees else 0.12
        if self.use_pynndescent:
            return self.search_index.query(X,n_neighbors,epsilon)
        else:
            if X.shape[0] == self._raw_data.shape[0]:
                return self.neighbor_graph
            else:
                if self.metric != 'cosine':
                    d,i = self.search_index.search(X,n_neighbors)
                else:
                    Xq = X.copy().astype(np.float32)
                    faiss.normalize_L2(Xq)
                    d,i = self.search_index.search(Xq,n_neighbors)
                if self.metric.lower() in ['cosine','ip', 'inner']:
                    d = 1-d
                return i,d
    
    def prepare(self):
        if self.use_pynndescent:
            self.search_index.prepare()
    
    def update(self,X):
        if self.use_pynndescent:
            self.search_index.update(X)
        else:
            self.search_index.add(X)
