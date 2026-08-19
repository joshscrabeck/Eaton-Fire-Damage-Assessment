"""
EATON FIRE - BUILDING DAMAGE ASSESSMENT
CLUSTERING FIRST + XGBOOST REFINEMENT
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MACHINE LEARNING IMPORTS
# ============================================================================
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ============================================================================
# CLUSTERING IMPORTS
# ============================================================================
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# ============================================================================
# CONFIGURATION
# ============================================================================

RAW_DIR = r'C:\Users\joshl\OneDrive\Documents\altadena_fire_map\docker_inator\data\sentinel2\raw'
BUILDINGS_GEOJSON = r'C:\Users\joshl\OneDrive\Documents\altadena_fire_map\docker_inator\data\overture\altadena_buildings.geojson'
BEFORE_IMAGE = os.path.join(RAW_DIR, 'altadena_before_10bands.tif')
AFTER_IMAGE = os.path.join(RAW_DIR, 'altadena_after_10bands.tif')
OUTPUT_DIR = r'C:\Users\joshl\OneDrive\Documents\altadena_fire_map\docker_inator\data\sentinel2\processed\analysis_results'

# ============================================================================
# CONFIGURATION
# ============================================================================
USE_SUBSET = False
SUBSET_SIZE = 1500
BUFFER_METERS = 500
MIN_PIXELS = 1
ALL_TOUCHED = False
BURN_THRESHOLD = 0.1

# Much lower thresholds for initial seed identification
SEED_DAMAGED = {
    'burn_percentage': 10,
    'dnbr_mean': 0.05
}

SEED_UNDAMAGED = {
    'burn_percentage': 2,
    'dnbr_mean': 0.01
}

# XGBoost configuration
MODEL_TYPE = 'xgboost'
CONFIDENCE_THRESHOLD = 0.4  # Lower threshold

# ============================================================================
# CLUSTERING CONFIGURATION
# ============================================================================
KNN_N_NEIGHBORS = 20
KMEANS_N_CLUSTERS = 8
CLUSTERING_METHOD = 'both'  # 'knn', 'kmeans', 'both', 'gmm'
CLUSTER_OVERRIDE_THRESHOLD = 0.4  # If cluster has >40% damaged, flag all

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# CLUSTERING-FIRST BUILDING DETECTOR
# ============================================================================

class ClusteringFirstDetector:
    def __init__(self, before_path, after_path, buildings_geojson,
                 use_subset=False, subset_size=1500, buffer_meters=500,
                 min_pixels=2, all_touched=False, burn_threshold=0.1,
                 seed_damaged=None, seed_undamaged=None,
                 model_type='xgboost', confidence_threshold=0.4,
                 knn_neighbors=20, kmeans_clusters=8,
                 clustering_method='both', cluster_override_threshold=0.4):
        
        self.before_path = before_path
        self.after_path = after_path
        self.buildings_geojson = buildings_geojson
        self.use_subset = use_subset
        self.subset_size = subset_size
        self.buffer_meters = buffer_meters
        self.min_pixels = min_pixels
        self.all_touched = all_touched
        self.burn_threshold = burn_threshold
        
        # Seed thresholds (used only to create initial labels)
        self.seed_damaged = seed_damaged or SEED_DAMAGED
        self.seed_undamaged = seed_undamaged or SEED_UNDAMAGED
        
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Clustering configuration
        self.knn_neighbors = knn_neighbors
        self.kmeans_clusters = kmeans_clusters
        self.clustering_method = clustering_method
        self.cluster_override_threshold = cluster_override_threshold
        
        self.before_data = None
        self.after_data = None
        self.buildings_gdf = None
        self.full_buildings_gdf = None
        self.clipped_before = None
        self.clipped_after = None
        self.clipped_transform = None
        
        self.band_mapping = {
            'blue': 1, 'green': 2, 'red': 3,
            'nir': 7, 'swir1': 9, 'swir2': 10,
        }
        
        self.results = None
        self.dnbr = None
        self.ndvi_change = None
        self.swir_change = None
        self.brightness_change = None
        self.burn_mask = None
        self.height = None
        self.width = None
        self.transform = None
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Clustering models
        self.knn_model = None
        self.kmeans_model = None
        self.gmm_model = None
        self.pca_model = None
        
    # ========================================================================
    # DATA LOADING
    # ========================================================================
    
    def load_and_clip_data(self):
        """Load and clip Sentinel-2 images"""
        print("\n" + "="*70)
        print("LOADING AND CLIPPING DATA")
        print("="*70)
        
        print("Loading building footprints...")
        self.full_buildings_gdf = gpd.read_file(self.buildings_geojson)
        print(f"  Loaded {len(self.full_buildings_gdf):,} building footprints")
        
        print("\nLoading Sentinel-2 images...")
        self.before_data = rasterio.open(self.before_path)
        self.after_data = rasterio.open(self.after_path)
        
        print(f"  Before: {self.before_data.shape}")
        print(f"  After: {self.after_data.shape}")
        print(f"  CRS: {self.before_data.crs}")
        
        if self.full_buildings_gdf.crs != self.before_data.crs:
            print(f"  Reprojecting buildings to {self.before_data.crs}")
            self.full_buildings_gdf = self.full_buildings_gdf.to_crs(self.before_data.crs)
        
        building_bounds = self.full_buildings_gdf.total_bounds
        print(f"\nBuilding bounds: ({building_bounds[0]:.0f}, {building_bounds[1]:.0f}) to ({building_bounds[2]:.0f}, {building_bounds[3]:.0f})")
        
        buffered_bounds = (
            building_bounds[0] - self.buffer_meters,
            building_bounds[1] - self.buffer_meters,
            building_bounds[2] + self.buffer_meters,
            building_bounds[3] + self.buffer_meters
        )
        print(f"Buffered bounds: ({buffered_bounds[0]:.0f}, {buffered_bounds[1]:.0f}) to ({buffered_bounds[2]:.0f}, {buffered_bounds[3]:.0f})")
        
        clip_box = box(*buffered_bounds)
        
        print("\nClipping before image to building extent...")
        self.clipped_before, self.clipped_transform = mask(
            self.before_data,
            [clip_box],
            crop=True
        )
        print(f"  Clipped before shape: {self.clipped_before.shape}")
        
        print("Clipping after image to building extent...")
        self.clipped_after, _ = mask(
            self.after_data,
            [clip_box],
            crop=True
        )
        print(f"  Clipped after shape: {self.clipped_after.shape}")
        
        self.height = self.clipped_before.shape[1]
        self.width = self.clipped_before.shape[2]
        self.transform = self.clipped_transform
        
        print(f"\n  Clipped dimensions: {self.height} x {self.width}")
        
        self.before_data.close()
        self.after_data.close()
        
        if self.use_subset:
            self.buildings_gdf = self.select_strategic_subset()
        else:
            self.buildings_gdf = self.full_buildings_gdf.copy()
            print(f"\nUsing all {len(self.buildings_gdf):,} buildings")
    
    def select_strategic_subset(self):
        """Select strategic subset"""
        print(f"\n  SUBSET MODE: Selecting {self.subset_size} buildings for testing")
        sample_size = min(self.subset_size, len(self.full_buildings_gdf))
        subset_gdf = self.full_buildings_gdf.sample(n=sample_size, random_state=42)
        print(f"  Final subset size: {len(subset_gdf):,} buildings")
        return subset_gdf
    
    # ========================================================================
    # SPECTRAL INDEX COMPUTATION
    # ========================================================================
    
    def compute_all_indices(self):
        """Compute ALL relevant indices"""
        print("\n" + "="*70)
        print("COMPUTING MULTIPLE INDICES")
        print("="*70)
        
        blue_idx = self.band_mapping['blue'] - 1
        green_idx = self.band_mapping['green'] - 1
        red_idx = self.band_mapping['red'] - 1
        nir_idx = self.band_mapping['nir'] - 1
        swir1_idx = self.band_mapping['swir1'] - 1
        swir2_idx = self.band_mapping['swir2'] - 1
        
        print("Extracting bands from clipped data...")
        before_blue = self.clipped_before[blue_idx].astype(np.float32)
        before_green = self.clipped_before[green_idx].astype(np.float32)
        before_red = self.clipped_before[red_idx].astype(np.float32)
        before_nir = self.clipped_before[nir_idx].astype(np.float32)
        before_swir1 = self.clipped_before[swir1_idx].astype(np.float32)
        before_swir2 = self.clipped_before[swir2_idx].astype(np.float32)
        
        after_blue = self.clipped_after[blue_idx].astype(np.float32)
        after_green = self.clipped_after[green_idx].astype(np.float32)
        after_red = self.clipped_after[red_idx].astype(np.float32)
        after_nir = self.clipped_after[nir_idx].astype(np.float32)
        after_swir1 = self.clipped_after[swir1_idx].astype(np.float32)
        after_swir2 = self.clipped_after[swir2_idx].astype(np.float32)
        
        eps = 1e-10
        
        # 1. dNBR
        print("  Computing dNBR...")
        before_nbr = (before_nir - before_swir2) / (before_nir + before_swir2 + eps)
        after_nbr = (after_nir - after_swir2) / (after_nir + after_swir2 + eps)
        self.dnbr = before_nbr - after_nbr
        
        # 2. NDVI Change
        print("  Computing NDVI Change...")
        before_ndvi = (before_nir - before_red) / (before_nir + before_red + eps)
        after_ndvi = (after_nir - after_red) / (after_nir + after_red + eps)
        self.ndvi_change = after_ndvi - before_ndvi
        
        # 3. SWIR Ratio Change
        print("  Computing SWIR Ratio Change...")
        before_swir_ratio = before_swir1 / (before_swir2 + eps)
        after_swir_ratio = after_swir1 / (after_swir2 + eps)
        self.swir_change = after_swir_ratio - before_swir_ratio
        
        # 4. Brightness Change
        print("  Computing Brightness Change...")
        before_brightness = (before_red + before_green + before_blue) / 3
        after_brightness = (after_red + after_green + after_blue) / 3
        self.brightness_change = after_brightness - before_brightness
        
        # Burn mask
        self.burn_mask = self.dnbr > self.burn_threshold
        
        burn_pixels = np.sum(self.burn_mask)
        total_pixels = self.dnbr.size
        
        print(f"\n  Burn Statistics:")
        print(f"    Total pixels: {total_pixels:,}")
        print(f"    Burn pixels (dNBR > {self.burn_threshold}): {burn_pixels:,}")
        print(f"    Burn percentage: {burn_pixels/total_pixels*100:.2f}%")
    
    # ========================================================================
    # BUILDING EXTRACTION
    # ========================================================================
    
    def extract_buildings(self):
        """Extract buildings and compute features"""
        print("\n" + "="*70)
        print("EXTRACTING BUILDING FEATURES")
        print("="*70)
        
        if len(self.buildings_gdf) == 0:
            print("  No buildings to process!")
            return None
        
        buildings_gdf = self.buildings_gdf.copy()
        buildings_gdf = buildings_gdf.reset_index(drop=True)
        buildings_gdf['raster_id'] = range(1, len(buildings_gdf) + 1)
        
        print(f"  Using {len(buildings_gdf):,} buildings with NO buffer")
        print(f"  Min pixels: {self.min_pixels}")
        
        shapes = [(row.geometry, row['raster_id']) for _, row in buildings_gdf.iterrows()]
        
        print("  Rasterizing building footprints...")
        
        building_id_raster = rasterize(
            shapes,
            out_shape=(self.height, self.width),
            transform=self.transform,
            dtype=np.int32,
            fill=0,
            all_touched=self.all_touched
        )
        
        unique_ids = np.unique(building_id_raster)
        unique_ids = unique_ids[unique_ids > 0]
        print(f"  {len(unique_ids):,} buildings rasterized")
        
        self.building_id_raster = building_id_raster
        self.filtered_gdf = buildings_gdf
        
        building_mask = building_id_raster > 0
        building_pixels = np.sum(building_mask)
        burn_overlap = np.sum(building_mask & self.burn_mask)
        
        print(f"\n  Building pixels: {building_pixels:,}")
        print(f"  Building pixels overlapping burn: {burn_overlap:,} ({burn_overlap/building_pixels*100:.2f}%)")
        
        print("\n  Processing buildings...")
        
        results = []
        
        for bid in tqdm(unique_ids, desc="  Analyzing buildings"):
            pixel_mask = (building_id_raster == bid)
            n_pixels = np.sum(pixel_mask)
            
            if n_pixels < self.min_pixels:
                continue
            
            building_row = buildings_gdf[buildings_gdf['raster_id'] == bid].iloc[0]
            
            # dNBR
            dnbr_vals = self.dnbr[pixel_mask]
            dnbr_vals = dnbr_vals[np.isfinite(dnbr_vals)]
            if len(dnbr_vals) < self.min_pixels:
                continue
            
            dnbr_mean = np.mean(dnbr_vals)
            dnbr_median = np.median(dnbr_vals)
            dnbr_max = np.max(dnbr_vals)
            dnbr_std = np.std(dnbr_vals)
            dnbr_p25 = np.percentile(dnbr_vals, 25)
            dnbr_p75 = np.percentile(dnbr_vals, 75)
            
            # Burn percentage
            building_burn_overlap = np.sum(pixel_mask & self.burn_mask)
            burn_percentage = building_burn_overlap / n_pixels if n_pixels > 0 else 0
            
            # NDVI change
            ndvi_vals = self.ndvi_change[pixel_mask]
            ndvi_vals = ndvi_vals[np.isfinite(ndvi_vals)]
            ndvi_mean = np.mean(ndvi_vals) if len(ndvi_vals) > 0 else 0
            ndvi_std = np.std(ndvi_vals) if len(ndvi_vals) > 0 else 0
            ndvi_min = np.min(ndvi_vals) if len(ndvi_vals) > 0 else 0
            
            # SWIR change
            swir_vals = self.swir_change[pixel_mask]
            swir_vals = swir_vals[np.isfinite(swir_vals)]
            swir_mean = np.mean(swir_vals) if len(swir_vals) > 0 else 0
            swir_std = np.std(swir_vals) if len(swir_vals) > 0 else 0
            swir_max = np.max(swir_vals) if len(swir_vals) > 0 else 0
            
            # Brightness change
            bright_vals = self.brightness_change[pixel_mask]
            bright_vals = bright_vals[np.isfinite(bright_vals)]
            bright_mean = np.mean(bright_vals) if len(bright_vals) > 0 else 0
            bright_std = np.std(bright_vals) if len(bright_vals) > 0 else 0
            
            results.append({
                'building_id': building_row.get('building_id', bid),
                'raster_id': bid,
                'n_pixels': n_pixels,
                'dnbr_mean': dnbr_mean,
                'dnbr_median': dnbr_median,
                'dnbr_max': dnbr_max,
                'dnbr_std': dnbr_std,
                'dnbr_p25': dnbr_p25,
                'dnbr_p75': dnbr_p75,
                'burn_pixels': building_burn_overlap,
                'burn_percentage': burn_percentage * 100,
                'ndvi_change_mean': ndvi_mean,
                'ndvi_change_std': ndvi_std,
                'ndvi_change_min': ndvi_min,
                'swir_change_mean': swir_mean,
                'swir_change_std': swir_std,
                'swir_change_max': swir_max,
                'brightness_change_mean': bright_mean,
                'brightness_change_std': bright_std,
            })
        
        print(f"\n  Processed {len(results):,} buildings")
        
        if len(results) == 0:
            print("  No buildings processed!")
            return None
        
        self.results = pd.DataFrame(results)
        return self.results
    
    # ========================================================================
    # CLUSTERING FIRST - MAIN APPROACH
    # ========================================================================
    
    def run_clustering_first(self):
        """
        Main approach: Clustering first to identify damaged patterns,
        then use XGBoost on remaining uncertain cases
        """
        print("\n" + "="*70)
        print("CLUSTERING-FIRST APPROACH")
        print("="*70)
        
        df = self.results
        
        # Step 1: Create seed labels using low thresholds
        print("\n  Step 1: Creating seed labels with low thresholds...")
        
        seed_damaged_mask = (
            (df['burn_percentage'] > self.seed_damaged['burn_percentage']) &
            (df['dnbr_mean'] > self.seed_damaged['dnbr_mean'])
        )
        
        seed_undamaged_mask = (
            (df['burn_percentage'] < self.seed_undamaged['burn_percentage']) &
            (df['dnbr_mean'] < self.seed_undamaged['dnbr_mean'])
        )
        
        df['seed_label'] = -1  # Unknown
        df.loc[seed_damaged_mask, 'seed_label'] = 1  # Damaged
        df.loc[seed_undamaged_mask, 'seed_label'] = 0  # Undamaged
        
        n_seed_damaged = np.sum(seed_damaged_mask)
        n_seed_undamaged = np.sum(seed_undamaged_mask)
        n_seed_unknown = len(df) - n_seed_damaged - n_seed_undamaged
        
        print(f"    Seed damaged: {n_seed_damaged:,} ({n_seed_damaged/len(df)*100:.1f}%)")
        print(f"    Seed undamaged: {n_seed_undamaged:,} ({n_seed_undamaged/len(df)*100:.1f}%)")
        print(f"    Seed unknown: {n_seed_unknown:,} ({n_seed_unknown/len(df)*100:.1f}%)")
        
        # Step 2: Apply clustering to identify damaged patterns
        print("\n  Step 2: Applying clustering to identify damaged patterns...")
        
        # Enhanced feature set for clustering
        cluster_features = [
            'dnbr_mean', 'dnbr_max', 'dnbr_std', 'dnbr_p25', 'dnbr_p75',
            'burn_percentage',
            'ndvi_change_mean', 'ndvi_change_min',
            'swir_change_mean', 'swir_change_max',
            'brightness_change_mean'
        ]
        
        # Prepare feature matrix
        X_all = df[cluster_features].values
        X_seed_damaged = df[seed_damaged_mask][cluster_features].values if n_seed_damaged > 0 else None
        X_seed_undamaged = df[seed_undamaged_mask][cluster_features].values if n_seed_undamaged > 0 else None
        
        # Standardize features
        scaler_cluster = StandardScaler()
        X_all_scaled = scaler_cluster.fit_transform(X_all)
        X_seed_damaged_scaled = scaler_cluster.transform(X_seed_damaged) if X_seed_damaged is not None else None
        X_seed_undamaged_scaled = scaler_cluster.transform(X_seed_undamaged) if X_seed_undamaged is not None else None
        
        # Initialize results columns
        df['cluster_label'] = -1
        df['cluster_damage_ratio'] = 0.0
        df['knn_distance_to_damaged'] = 0.0
        df['knn_distance_to_undamaged'] = 0.0
        df['clustering_damaged'] = 0
        df['clustering_confidence'] = 0.0
        df['found_by_ml'] = False
        
        # Method 1: K-means clustering
        if self.clustering_method in ['kmeans', 'both']:
            print("\n    K-means Clustering...")
            
            # Find optimal number of clusters
            n_clusters = self.kmeans_clusters
            if n_clusters <= 1:
                inertias = []
                for k in range(2, 10):
                    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans_test.fit(X_all_scaled)
                    inertias.append(kmeans_test.inertia_)
                diffs = np.diff(inertias)
                n_clusters = np.argmin(diffs) + 2
            
            # Apply K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_all_scaled)
            df['cluster_label'] = cluster_labels
            
            # Calculate damage ratio per cluster
            cluster_damage_ratios = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_damaged = np.sum(cluster_mask & seed_damaged_mask)
                    cluster_total = np.sum(cluster_mask)
                    damage_ratio = cluster_damaged / cluster_total if cluster_total > 0 else 0
                    cluster_damage_ratios[cluster_id] = damage_ratio
                    df.loc[cluster_mask, 'cluster_damage_ratio'] = damage_ratio
            
            # Identify damaged clusters
            high_damage_clusters = [c for c, r in cluster_damage_ratios.items() 
                                   if r > self.cluster_override_threshold]
            
            print(f"    Found {n_clusters} clusters")
            print(f"    High-damage clusters (> {self.cluster_override_threshold:.0%}): {len(high_damage_clusters)}")
            
            # Assign clustering predictions
            for cluster_id in high_damage_clusters:
                cluster_mask = cluster_labels == cluster_id
                df.loc[cluster_mask, 'clustering_damaged'] = 1
                df.loc[cluster_mask, 'clustering_confidence'] = cluster_damage_ratios[cluster_id]
        
        # Method 2: K-NN analysis
        if self.clustering_method in ['knn', 'both'] and X_seed_damaged_scaled is not None:
            print("\n    K-NN Analysis...")
            
            # K-NN to damaged seeds
            knn_damaged = NearestNeighbors(n_neighbors=min(self.knn_neighbors, len(X_seed_damaged_scaled)))
            knn_damaged.fit(X_seed_damaged_scaled)
            distances_damaged, _ = knn_damaged.kneighbors(X_all_scaled)
            avg_dist_damaged = np.mean(distances_damaged, axis=1)
            df['knn_distance_to_damaged'] = avg_dist_damaged
            
            # K-NN to undamaged seeds (if available)
            if X_seed_undamaged_scaled is not None and len(X_seed_undamaged_scaled) > 0:
                knn_undamaged = NearestNeighbors(n_neighbors=min(self.knn_neighbors, len(X_seed_undamaged_scaled)))
                knn_undamaged.fit(X_seed_undamaged_scaled)
                distances_undamaged, _ = knn_undamaged.kneighbors(X_all_scaled)
                avg_dist_undamaged = np.mean(distances_undamaged, axis=1)
                df['knn_distance_to_undamaged'] = avg_dist_undamaged
                
                # K-NN score: lower distance to damaged = more likely damaged
                knn_score = 1 / (avg_dist_damaged + 0.001) / (1 / (avg_dist_damaged + 0.001) + 1 / (avg_dist_undamaged + 0.001))
            else:
                # Only damaged reference - use percentile threshold
                damaged_dist_threshold = np.percentile(avg_dist_damaged[seed_damaged_mask], 30)
                knn_score = 1 - (avg_dist_damaged / (damaged_dist_threshold * 2))
                knn_score = np.clip(knn_score, 0, 1)
            
            # Update clustering predictions with K-NN
            knn_damaged_mask = (knn_score > 0.5) & (df['clustering_damaged'] == 0)
            df.loc[knn_damaged_mask, 'clustering_damaged'] = 1
            df.loc[knn_damaged_mask, 'clustering_confidence'] = knn_score[knn_damaged_mask]
            
            print(f"    K-NN flagged {np.sum(knn_damaged_mask):,} additional buildings as damaged")
        
        # Method 3: GMM for outlier detection
        if self.clustering_method == 'gmm':
            print("\n    GMM Outlier Detection...")
            
            gmm = GaussianMixture(n_components=min(n_clusters, 5), random_state=42)
            gmm.fit(X_all_scaled)
            log_likelihood = gmm.score_samples(X_all_scaled)
            
            # Outliers with low likelihood might be damaged
            threshold = np.percentile(log_likelihood[seed_damaged_mask], 20)
            outlier_mask = (log_likelihood < threshold) & (df['clustering_damaged'] == 0)
            df.loc[outlier_mask, 'clustering_damaged'] = 1
            df.loc[outlier_mask, 'clustering_confidence'] = 0.5
            
            print(f"    GMM flagged {np.sum(outlier_mask):,} outlier buildings as damaged")
        
        # Step 3: Consolidate clustering results
        print("\n  Step 3: Consolidating clustering results...")
        
        n_clustering_damaged = np.sum(df['clustering_damaged'] == 1)
        n_clustering_damaged_already_seed = np.sum((df['clustering_damaged'] == 1) & seed_damaged_mask)
        n_clustering_damaged_new = n_clustering_damaged - n_clustering_damaged_already_seed
        
        print(f"    Clustering damaged total: {n_clustering_damaged:,}")
        print(f"      Already in seed damaged: {n_clustering_damaged_already_seed:,}")
        print(f"      NEW buildings found: {n_clustering_damaged_new:,}")
        
        # Step 4: Train XGBoost on remaining unknown cases
        print("\n  Step 4: Training XGBoost on remaining unknown cases...")
        
        # Identify cases that need ML refinement
        # Use clustering-damaged as positive labels for training
        ml_train_mask = (df['clustering_damaged'] == 1) | (seed_undamaged_mask)
        ml_train = df[ml_train_mask].copy()
        
        if len(ml_train) > 0:
            # Features for ML
            self.feature_names = [
                'n_pixels',
                'dnbr_mean', 'dnbr_median', 'dnbr_max', 'dnbr_std', 'dnbr_p25', 'dnbr_p75',
                'burn_percentage',
                'ndvi_change_mean', 'ndvi_change_std', 'ndvi_change_min',
                'swir_change_mean', 'swir_change_std', 'swir_change_max',
                'brightness_change_mean', 'brightness_change_std'
            ]
            
            X_train = ml_train[self.feature_names].values
            y_train = ml_train['clustering_damaged'].values
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train XGBoost
            self.model = XGBClassifier(
                n_estimators=150, 
                max_depth=6, 
                learning_rate=0.1,
                random_state=42, 
                n_jobs=-1, 
                eval_metric='logloss',
                scale_pos_weight=(len(y_train) - np.sum(y_train)) / np.sum(y_train) if np.sum(y_train) > 0 else 1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Predict on all buildings
            X_all_ml = df[self.feature_names].values
            X_all_ml_scaled = self.scaler.transform(X_all_ml)
            
            ml_proba = self.model.predict_proba(X_all_ml_scaled)[:, 1]
            df['ml_probability'] = ml_proba
            df['ml_prediction'] = self.model.predict(X_all_ml_scaled)
            
            # Feature importance
            importance = self.model.feature_importances_
            print(f"\n    Top 5 important features:")
            sorted_idx = np.argsort(importance)[::-1]
            for i in range(min(5, len(sorted_idx))):
                idx = sorted_idx[i]
                print(f"      {self.feature_names[idx]}: {importance[idx]:.4f}")
        else:
            print("    No training data available for ML")
            df['ml_probability'] = 0
            df['ml_prediction'] = 0
        
        # Step 5: Final classification - combine clustering and ML
        print("\n  Step 5: Final classification...")
        
        # Start with clustering results
        df['final_damaged'] = df['clustering_damaged']
        
        # Use ML to find additional damaged buildings not found by clustering
        # Only apply ML to buildings with high probability AND not already classified
        ml_new_mask = (
            (df['clustering_damaged'] == 0) &
            (df['ml_probability'] > self.confidence_threshold)
        )
        df.loc[ml_new_mask, 'final_damaged'] = 1
        df.loc[ml_new_mask, 'found_by_ml'] = True
        
        # Severity assignment
        df['severity'] = 0
        df.loc[df['final_damaged'] == 1, 'severity'] = 1  # Default low
        
        # Higher severity for high confidence cases
        high_conf_mask = (
            (df['final_damaged'] == 1) & 
            ((df['clustering_confidence'] > 0.7) | (df['ml_probability'] > 0.7))
        )
        df.loc[high_conf_mask, 'severity'] = 3
        
        med_conf_mask = (
            (df['final_damaged'] == 1) & 
            ((df['clustering_confidence'] > 0.5) | (df['ml_probability'] > 0.5)) &
            ~high_conf_mask
        )
        df.loc[med_conf_mask, 'severity'] = 2
        
        # High confidence flag
        df['high_confidence'] = (
            (df['severity'] == 3) |
            ((df['final_damaged'] == 1) & (df['clustering_confidence'] > 0.6)) |
            ((df['final_damaged'] == 1) & (df['ml_probability'] > 0.7))
        )
        
        # Summary
        total = len(df)
        final_damaged = np.sum(df['final_damaged'] == 1)
        high_conf_damaged = np.sum((df['final_damaged'] == 1) & (df['high_confidence'] == True))
        cluster_found = np.sum((df['final_damaged'] == 1) & (df['clustering_damaged'] == 1))
        ml_found = np.sum((df['final_damaged'] == 1) & (df['found_by_ml'] == True))
        
        print(f"\n  Final Classification Summary:")
        print(f"    Total buildings: {total:,}")
        print(f"    Damaged: {final_damaged:,} ({final_damaged/total*100:.1f}%)")
        print(f"    High confidence damaged: {high_conf_damaged:,} ({high_conf_damaged/total*100:.1f}%)")
        print(f"    Found by clustering: {cluster_found:,} ({cluster_found/total*100:.1f}%)")
        print(f"    Found by ML refinement: {ml_found:,} ({ml_found/total*100:.1f}%)")
        
        self.results = df
        
        # Create visualizations
        self.create_visualizations()
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n  Creating visualizations...")
        
        df = self.results
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Damage status breakdown
        ax = axes[0, 0]
        counts = df['final_damaged'].value_counts()
        colors = ['green', 'red']
        ax.bar(['Undamaged', 'Damaged'], 
               [counts.get(0, 0), counts.get(1, 0)],
               color=['green', 'red'])
        
        # Overlay clustering vs ML found
        cluster_count = np.sum((df['final_damaged'] == 1) & (df['clustering_damaged'] == 1))
        ml_count = np.sum((df['final_damaged'] == 1) & (df['found_by_ml'] == True))
        if cluster_count > 0 or ml_count > 0:
            ax.bar(['Damaged'], [cluster_count], color='blue', alpha=0.5, label='Clustering')
            ax.bar(['Damaged'], [ml_count], bottom=[cluster_count], color='purple', alpha=0.5, label='ML Refinement')
        ax.set_title('Final Damage Assessment')
        ax.set_xlabel('Status')
        ax.set_ylabel('Count')
        ax.legend()
        
        # 2. Burn percentage vs dNBR with classification
        ax = axes[0, 1]
        ax.scatter(df[df['final_damaged'] == 0]['burn_percentage'], 
                  df[df['final_damaged'] == 0]['dnbr_mean'],
                  c='green', label='Undamaged', alpha=0.3, s=5)
        ax.scatter(df[df['final_damaged'] == 1]['burn_percentage'], 
                  df[df['final_damaged'] == 1]['dnbr_mean'],
                  c='red', label='Damaged', alpha=0.5, s=5)
        
        # Highlight clustering-found buildings
        cluster_mask = (df['clustering_damaged'] == 1) & (df['final_damaged'] == 1)
        if np.sum(cluster_mask) > 0:
            ax.scatter(df.loc[cluster_mask, 'burn_percentage'], 
                      df.loc[cluster_mask, 'dnbr_mean'],
                      c='blue', marker='*', s=40, label='Found by Clustering')
        
        ax.set_title('Burn Percentage vs dNBR')
        ax.set_xlabel('Burn Percentage (%)')
        ax.set_ylabel('dNBR Mean')
        ax.legend()
        
        # 3. Severity breakdown - FIXED
        ax = axes[0, 2]
        severity_counts = df['severity'].value_counts().sort_index()
        labels = ['Undamaged', 'Low', 'Moderate', 'High']
        colors_sev = ['green', 'yellow', 'orange', 'red']
        
        # Get counts for all 4 categories, default to 0 if missing
        counts_sev = [severity_counts.get(i, 0) for i in range(4)]
        ax.bar(labels, counts_sev, color=colors_sev)
        ax.set_title('Severity Breakdown')
        ax.set_xlabel('Severity')
        ax.set_ylabel('Count')
        
        # 4. ML Probability distribution by final class
        ax = axes[1, 0]
        ax.hist(df[df['final_damaged'] == 1]['ml_probability'].values, 
                bins=20, alpha=0.7, color='red', label='Damaged')
        ax.hist(df[df['final_damaged'] == 0]['ml_probability'].values, 
                bins=20, alpha=0.7, color='green', label='Undamaged')
        ax.axvline(x=self.confidence_threshold, color='purple', linestyle='--', 
                  label=f'ML threshold ({self.confidence_threshold})')
        ax.set_title('ML Probability Distribution')
        ax.set_xlabel('ML Probability')
        ax.set_ylabel('Count')
        ax.legend()
        
        # 5. Clustering confidence
        ax = axes[1, 1]
        ax.hist(df[df['final_damaged'] == 1]['clustering_confidence'].values, 
                bins=20, alpha=0.7, color='blue', label='Damaged')
        ax.hist(df[df['final_damaged'] == 0]['clustering_confidence'].values, 
                bins=20, alpha=0.7, color='gray', label='Undamaged')
        ax.set_title('Clustering Confidence Distribution')
        ax.set_xlabel('Clustering Confidence')
        ax.set_ylabel('Count')
        ax.legend()
        
        # 6. Discovery method pie chart
        ax = axes[1, 2]
        damaged_df = df[df['final_damaged'] == 1]
        if len(damaged_df) > 0:
            cluster_only = np.sum((damaged_df['clustering_damaged'] == 1) & (damaged_df['found_by_ml'] == False))
            ml_only = np.sum((damaged_df['clustering_damaged'] == 0) & (damaged_df['found_by_ml'] == True))
            both = np.sum((damaged_df['clustering_damaged'] == 1) & (damaged_df['found_by_ml'] == True))
            
            methods = []
            if cluster_only > 0:
                methods.append(('Clustering Only', cluster_only))
            if ml_only > 0:
                methods.append(('ML Only', ml_only))
            if both > 0:
                methods.append(('Both', both))
            
            if methods:
                labels_pie, sizes = zip(*methods)
                ax.pie(sizes, labels=labels_pie, autopct='%1.1f%%', 
                      colors=['blue', 'purple', 'orange'])
        ax.set_title('Discovery Method for Damaged Buildings')
        
        plt.tight_layout()
        viz_path = os.path.join(OUTPUT_DIR, 'clustering_first_analysis.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved to: {viz_path}")
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def create_report(self):
        """Create comprehensive report"""
        print("\n" + "="*70)
        print("DAMAGE ASSESSMENT REPORT - CLUSTERING FIRST APPROACH")
        print(f"MODEL: {self.model_type.upper()}")
        print(f"CLUSTERING: {self.clustering_method.upper()}")
        print(f"CONFIDENCE THRESHOLD: {self.confidence_threshold}")
        if self.use_subset:
            print(f"SUBSET TEST - {self.subset_size} buildings")
        else:
            print(f"FULL ANALYSIS - {len(self.full_buildings_gdf):,} buildings")
        print("="*70)
        
        df = self.results
        
        total = len(df)
        damaged = np.sum(df['final_damaged'] == 1)
        high_conf = np.sum((df['final_damaged'] == 1) & (df['high_confidence'] == True))
        cluster_found = np.sum((df['final_damaged'] == 1) & (df['clustering_damaged'] == 1))
        ml_found = np.sum((df['final_damaged'] == 1) & (df['found_by_ml'] == True))
        
        print(f"\nTotal Buildings Analyzed: {total:,}")
        print(f"  Damaged: {damaged:,} ({damaged/total*100:.1f}%)")
        print(f"  High Confidence Damaged: {high_conf:,} ({high_conf/total*100:.1f}%)")
        print(f"  Found by Clustering: {cluster_found:,} ({cluster_found/total*100:.1f}%)")
        print(f"  Found by ML Refinement: {ml_found:,} ({ml_found/total*100:.1f}%)")
        
        # Severity breakdown
        print(f"\nSeverity Breakdown:")
        severity_counts = df['severity'].value_counts().sort_index()
        for sev, label in [(0, 'Undamaged'), (1, 'Low'), (2, 'Moderate'), (3, 'High')]:
            count = severity_counts.get(sev, 0)
            pct = count / len(df) * 100
            print(f"  {label}: {count:,} ({pct:.1f}%)")
        
        # Save CSV
        output_csv = os.path.join(OUTPUT_DIR, f'building_damage_clustering_first_{self.model_type}_{len(df)}.csv')
        df.to_csv(output_csv, index=False)
        print(f"\n  Results saved to: {output_csv}")
        
        # Export GeoJSON
        self.export_geojson(df)
    
    def export_geojson(self, df):
        """Export results as GeoJSON"""
        print("  Exporting GeoJSON...")
        
        gdf_export = self.filtered_gdf.copy()
        gdf_export = gdf_export.merge(
            df[['raster_id', 'final_damaged', 'severity', 'ml_probability',
                'burn_percentage', 'dnbr_mean', 'high_confidence',
                'clustering_damaged', 'cluster_damage_ratio', 
                'knn_distance_to_damaged', 'clustering_confidence']],
            on='raster_id',
            how='left'
        )
        
        # Fill NaN
        gdf_export['final_damaged'] = gdf_export['final_damaged'].fillna(0).astype(int)
        gdf_export['severity'] = gdf_export['severity'].fillna(0).astype(int)
        gdf_export['ml_probability'] = gdf_export['ml_probability'].fillna(0)
        gdf_export['burn_percentage'] = gdf_export['burn_percentage'].fillna(0)
        gdf_export['dnbr_mean'] = gdf_export['dnbr_mean'].fillna(0)
        gdf_export['high_confidence'] = gdf_export['high_confidence'].fillna(False)
        gdf_export['clustering_damaged'] = gdf_export['clustering_damaged'].fillna(0).astype(int)
        gdf_export['cluster_damage_ratio'] = gdf_export['cluster_damage_ratio'].fillna(0)
        gdf_export['clustering_confidence'] = gdf_export['clustering_confidence'].fillna(0)
        gdf_export['knn_distance_to_damaged'] = gdf_export['knn_distance_to_damaged'].fillna(0)
        
        # Labels
        status_map = {0: 'Undamaged', 1: 'Low', 2: 'Moderate', 3: 'High'}
        gdf_export['damage_status'] = gdf_export['severity'].map(status_map)
        gdf_export['confidence_label'] = gdf_export['high_confidence'].map(
            {True: 'High Confidence', False: 'Low Confidence'}
        )
        gdf_export['discovery_method'] = 'unknown'
        gdf_export.loc[gdf_export['clustering_damaged'] == 1, 'discovery_method'] = 'clustering'
        gdf_export.loc[(gdf_export['clustering_damaged'] == 0) & (gdf_export['final_damaged'] == 1), 'discovery_method'] = 'ml_refinement'
        
        # Filter to damaged only
        gdf_export = gdf_export[gdf_export['final_damaged'] == 1]
        
        output_geojson = os.path.join(OUTPUT_DIR, f'damaged_buildings_clustering_first_{self.model_type}.geojson')
        gdf_export.to_file(output_geojson, driver='GeoJSON')
        print(f"  GeoJSON saved to: {output_geojson}")
    
    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*70)
        print("EATON FIRE - BUILDING DAMAGE ASSESSMENT")
        print("CLUSTERING FIRST APPROACH")
        print(f"MODEL: {self.model_type.upper()}")
        print(f"CLUSTERING: {self.clustering_method.upper()}")
        if self.use_subset:
            print(f"SUBSET TEST MODE - {self.subset_size} buildings")
        else:
            print("FULL ANALYSIS - ALL BUILDINGS")
        print("="*70)
        
        # Step 1-3: Load, compute indices, extract features
        self.load_and_clip_data()
        self.compute_all_indices()
        self.extract_buildings()
        
        if self.results is not None:
            # Step 4: Run clustering-first analysis
            self.run_clustering_first()
            
            # Step 5: Create report
            self.create_report()
        else:
            print("\n  No results produced.")
        
        print("\n" + "="*70)
        print("ASSESSMENT COMPLETE!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    detector = ClusteringFirstDetector(
        BEFORE_IMAGE,
        AFTER_IMAGE,
        BUILDINGS_GEOJSON,
        use_subset=USE_SUBSET,
        subset_size=SUBSET_SIZE,
        buffer_meters=BUFFER_METERS,
        min_pixels=MIN_PIXELS,
        all_touched=ALL_TOUCHED,
        burn_threshold=BURN_THRESHOLD,
        seed_damaged=SEED_DAMAGED,
        seed_undamaged=SEED_UNDAMAGED,
        model_type=MODEL_TYPE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        knn_neighbors=KNN_N_NEIGHBORS,
        kmeans_clusters=KMEANS_N_CLUSTERS,
        clustering_method=CLUSTERING_METHOD,
        cluster_override_threshold=CLUSTER_OVERRIDE_THRESHOLD
    )
    
    detector.run_analysis()
    return detector


if __name__ == "__main__":
    results = main()