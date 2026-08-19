"""
data_downloader_10bands_fixed.py - Prioritizes Correct Tile
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import requests
import json
from shapely.geometry import box, Point
import geopandas as gpd
import numpy as np
import warnings
import zipfile
import io
import os
import shutil
from rasterio.warp import reproject, Resampling

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EatonFireDataDownloader:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "data"
        self.sentinel_dir = self.output_dir / "sentinel2" / "raw"
        self.overture_dir = self.output_dir / "overture"
        
        self.sentinel_dir.mkdir(parents=True, exist_ok=True)
        self.overture_dir.mkdir(parents=True, exist_ok=True)
        
        # Eaton Fire bounding box (Altadena foothills)
        self.altadena_bbox = (-118.12, 34.16, -118.04, 34.23)
        
        # Eaton Fire dates
        self.fire_start = datetime(2025, 1, 7)
        self.fire_end = datetime(2025, 1, 31)
        
        # All 10 bands
        self.band_order = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
        self.band_names = {
            'B02': 'Blue', 'B03': 'Green', 'B04': 'Red',
            'B05': 'Red Edge 1', 'B06': 'Red Edge 2', 'B07': 'Red Edge 3',
            'B08': 'NIR', 'B8A': 'Narrow NIR', 'B11': 'SWIR1', 'B12': 'SWIR2'
        }
        self.bands_10m = ['B02', 'B03', 'B04', 'B08']
        self.bands_20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        
        # CRITICAL: The correct tile for Altadena
        self.correct_tiles = ['T11SLT', 'T11SLU']  # Altadena is in T11SLT
        
        logger.info("=" * 60)
        logger.info("EATON FIRE - DATA DOWNLOADER (CORRECTED)")
        logger.info("=" * 60)
        logger.info(f"Target tiles: {self.correct_tiles}")
    
    def download_all(self):
        logger.info("\n" + "=" * 60)
        logger.info("STARTING DATA DOWNLOAD")
        logger.info("=" * 60)
        
        self.download_sentinel_all_bands()
        self.download_overture_buildings()
        
        logger.info("\n" + "=" * 60)
        logger.info("DATA DOWNLOAD COMPLETE")
        logger.info("=" * 60)
    
    def download_sentinel_all_bands(self):
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOADING SENTINEL-2 IMAGES")
        logger.info("=" * 60)
        
        before_merged = self.sentinel_dir / "altadena_before_10bands.tif"
        after_merged = self.sentinel_dir / "altadena_after_10bands.tif"
        
        # Check if files exist and are valid (and from correct tile)
        if before_merged.exists() and after_merged.exists():
            try:
                with rasterio.open(before_merged) as src:
                    before_count = src.count
                with rasterio.open(after_merged) as src:
                    after_count = src.count
                
                if before_count == 10 and after_count == 10:
                    logger.info("✅ 10-band files already exist")
                    return
            except:
                pass
        
        try:
            import pystac_client
            import planetary_computer
            from pystac_client import Client
            
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=planetary_computer.sign_inplace,
            )
            
            # Pre-fire: 60-90 days before
            before_start = self.fire_start - timedelta(days=90)
            before_end = self.fire_start - timedelta(days=7)
            
            # Post-fire: February 1 - March 15
            after_start = datetime(2025, 2, 1)
            after_end = datetime(2025, 3, 15)
            
            logger.info(f"\nPre-fire images: {before_start.date()} to {before_end.date()}")
            logger.info(f"Post-fire images: {after_start.date()} to {after_end.date()}")
            
            # Download both pre and post images
            for date_start, date_end, label in [
                (before_start, before_end, "before"),
                (after_start, after_end, "after")
            ]:
                output_file = self.sentinel_dir / f"altadena_{label}_10bands.tif"
                if output_file.exists():
                    logger.info(f"File already exists: {output_file}")
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Searching for {label}-fire imagery...")
                logger.info(f"Date range: {date_start.date()} to {date_end.date()}")
                logger.info(f"{'='*60}")
                
                # Search with low cloud cover
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=self.altadena_bbox,
                    datetime=f"{date_start.strftime('%Y-%m-%d')}/{date_end.strftime('%Y-%m-%d')}",
                    query={"eo:cloud_cover": {"lt": 20}},  # Allow slightly more cloud to find correct tile
                    max_items=50
                )
                
                items = list(search.get_items())
                logger.info(f"Found {len(items)} items")
                
                if not items:
                    logger.warning(f"No items found for {label}-fire period")
                    continue
                
                # CRITICAL FIX: Sort by tile priority FIRST, then cloud cover
                # We want T11SLT > T11SLU > anything else
                def item_priority(item):
                    tile = item.id.split('_')[-2] if '_' in item.id else 'unknown'
                    cloud = item.properties.get('eo:cloud_cover', 100)
                    
                    # Priority: 0 = T11SLT, 1 = T11SLU, 2 = others
                    if tile == 'T11SLT':
                        priority = 0
                    elif tile == 'T11SLU':
                        priority = 1
                    else:
                        priority = 2
                    
                    # Return tuple (priority, cloud_cover) for sorting
                    return (priority, cloud)
                
                items_sorted = sorted(items, key=item_priority)
                
                # Show top 10 items with their priority
                logger.info("\nAvailable images (sorted by tile priority then cloud cover):")
                for i, item in enumerate(items_sorted[:10]):
                    cloud = item.properties.get('eo:cloud_cover', 'unknown')
                    date = item.properties.get('datetime', 'unknown')[:10]
                    tile = item.id.split('_')[-2] if '_' in item.id else 'unknown'
                    priority = "✓" if tile in self.correct_tiles else " "
                    logger.info(f"  {i+1}. {priority} Date: {date}, Cloud: {cloud:.2f}%, Tile: {tile}")
                
                # Select the best item (prioritizing correct tile)
                best_item = None
                for item in items_sorted:
                    tile = item.id.split('_')[-2] if '_' in item.id else 'unknown'
                    cloud = item.properties.get('eo:cloud_cover', 100)
                    
                    # If it's the correct tile and cloud < 20%, use it
                    if tile in self.correct_tiles and cloud < 20:
                        best_item = item
                        logger.info(f"\n✅ Selected {label} image: {item.id}")
                        logger.info(f"   Tile: {tile}, Cloud: {cloud:.2f}%")
                        logger.info(f"   Date: {item.properties.get('datetime', 'unknown')}")
                        break
                
                # Fallback: if no correct tile, use best available
                if best_item is None and items_sorted:
                    best_item = items_sorted[0]
                    tile = best_item.id.split('_')[-2] if '_' in best_item.id else 'unknown'
                    cloud = best_item.properties.get('eo:cloud_cover', 'unknown')
                    logger.warning(f"\n⚠️ No correct tile found, using {tile} (cloud: {cloud:.2f}%)")
                
                if best_item:
                    self._download_and_merge_bands_fixed(best_item, label)
                else:
                    logger.warning(f"No suitable image found for {label}")
                    
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.info("Please install: pip install pystac-client planetary-computer")
        except Exception as e:
            logger.error(f"Error downloading Sentinel-2 data: {e}")
            import traceback
            traceback.print_exc()
    
    def _download_and_merge_bands_fixed(self, item, label):
        """Download all bands with proper resampling"""
        logger.info(f"\nDownloading and merging bands for {label}...")
        
        band_data = {}
        reference_shape = None
        reference_transform = None
        reference_crs = None
        
        # Download all bands
        for band_name in self.band_order:
            if band_name in item.assets:
                try:
                    asset = item.assets[band_name]
                    logger.info(f"Downloading band {band_name} ({self.band_names[band_name]})...")
                    
                    response = requests.get(asset.href, timeout=120)
                    if response.status_code == 200:
                        with rasterio.open(io.BytesIO(response.content)) as src:
                            band_data[band_name] = src.read(1)
                            logger.info(f"  ✅ Downloaded {band_name}, shape: {src.shape}")
                            
                            if band_name in self.bands_10m and reference_shape is None:
                                reference_shape = src.shape
                                reference_transform = src.transform
                                reference_crs = src.crs
                    else:
                        logger.warning(f"  Failed to download {band_name}")
                except Exception as e:
                    logger.error(f"Error downloading {band_name}: {e}")
            else:
                logger.warning(f"Band {band_name} not found in assets")
        
        if len(band_data) < 4:
            logger.error(f"Only downloaded {len(band_data)} bands, need at least 4")
            return
        
        if reference_shape is None:
            logger.error("No 10m band found!")
            return
        
        # Resample 20m bands to 10m
        logger.info(f"\nResampling bands to match reference shape: {reference_shape}")
        
        dst_transform = reference_transform
        dst_shape = reference_shape
        
        resampled_bands = []
        for band_name in self.band_order:
            if band_name in band_data:
                data = band_data[band_name]
                
                if band_name in self.bands_20m:
                    logger.info(f"  Resampling {band_name} from {data.shape} to {dst_shape}...")
                    
                    src_transform = reference_transform * reference_transform.scale(2, 2)
                    
                    resampled = np.zeros(dst_shape, dtype=np.float32)
                    
                    reproject(
                        source=data,
                        destination=resampled,
                        src_transform=src_transform,
                        src_crs=reference_crs,
                        src_nodata=None,
                        dst_transform=dst_transform,
                        dst_crs=reference_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear
                    )
                    
                    resampled_bands.append(resampled)
                else:
                    resampled_bands.append(data.astype(np.float32))
            else:
                logger.warning(f"  Band {band_name} missing, creating zeros")
                resampled_bands.append(np.zeros(dst_shape, dtype=np.float32))
        
        # Save with BIGTIFF support
        if len(resampled_bands) == len(self.band_order):
            output_file = self.sentinel_dir / f"altadena_{label}_10bands.tif"
            logger.info(f"\nSaving merged 10-band file to: {output_file}")
            
            stack = np.stack(resampled_bands)
            
            # CRITICAL FIX: Use BIGTIFF for large files (> 4GB)
            meta = {
                'driver': 'GTiff',
                'height': dst_shape[0],
                'width': dst_shape[1],
                'count': len(self.band_order),
                'dtype': 'float32',
                'crs': reference_crs,
                'transform': dst_transform,
                'compress': 'DEFLATE',
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'nodata': np.nan,
                'BIGTIFF': 'YES'  # CRITICAL: Allows files > 4GB
            }
            
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band in enumerate(stack, 1):
                    dst.write(band, i)
            
            logger.info(f"✅ Saved 10-band file: {output_file}")
            file_size = output_file.stat().st_size / (1024 * 1024 * 1024)
            logger.info(f"   Size: {file_size:.2f} GB, Shape: {stack.shape}, Bands: {len(self.band_order)}")
        else:
            logger.error(f"Not enough bands: {len(resampled_bands)}/{len(self.band_order)}")
    
    def download_overture_buildings(self):
        logger.info("\n" + "=" * 60)
        logger.info("CHECKING OVERTURE BUILDINGS")
        logger.info("=" * 60)
        
        output_file = self.overture_dir / "altadena_buildings.geojson"
        
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)
            if file_size > 1:
                try:
                    gdf = gpd.read_file(output_file)
                    logger.info(f"Found {len(gdf)} buildings")
                    return output_file
                except:
                    pass
        
        # Check original location
        original_overture = self.project_root.parent / "data" / "overture" / "altadena_buildings.geojson"
        if original_overture.exists():
            logger.info(f"Copying buildings from: {original_overture}")
            shutil.copy2(original_overture, output_file)
            gdf = gpd.read_file(output_file)
            logger.info(f"Found {len(gdf)} buildings")
            return output_file
        
        logger.warning("Building file not found!")
        return None


def main():
    logger.info("=" * 60)
    logger.info("EATON FIRE - DATA DOWNLOADER (CORRECTED)")
    logger.info("Prioritizing T11SLT/T11SLU tiles for Altadena")
    logger.info("=" * 60)
    
    downloader = EatonFireDataDownloader()
    downloader.download_all()


if __name__ == "__main__":
    main()