"""
EATON FIRE - BUILDING DAMAGE WEB MAP
Simplified version - Building footprints only
FIXED: Handles missing severity column
BASEMAP: OpenStreetMap
"""

import os
import pandas as pd
import numpy as np
import folium
from folium.plugins import Fullscreen, MousePosition, MeasureControl
import geopandas as gpd
from pathlib import Path
import webbrowser
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path(r'C:\Users\joshl\OneDrive\Documents\altadena_fire_map\docker_inator\data\sentinel2\processed\analysis_results')

# LATEST OUTPUT - Clustering First with min_pixels=1
BUILDINGS_CSV = DATA_DIR / 'building_damage_clustering_first_xgboost_37291.csv'
BUILDINGS_GEOJSON = DATA_DIR / 'damaged_buildings_clustering_first_xgboost.geojson'

OUTPUT_FILE = Path('eaton_fire_damage_map.html')

# Altadena center
CENTER_LAT = 34.19
CENTER_LON = -118.13

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load and merge damage data with building footprints"""
    
    # Load CSV
    if not BUILDINGS_CSV.exists():
        print(f"CSV not found: {BUILDINGS_CSV}")
        return None
    
    df = pd.read_csv(BUILDINGS_CSV)
    print(f"Loaded {len(df):,} records from CSV")
    
    # Load GeoJSON
    if not BUILDINGS_GEOJSON.exists():
        print(f"GeoJSON not found: {BUILDINGS_GEOJSON}")
        return None
    
    gdf = gpd.read_file(BUILDINGS_GEOJSON)
    print(f"Loaded {len(gdf):,} damaged building footprints from GeoJSON")
    
    # Check what columns we have
    print(f"  GeoJSON columns: {gdf.columns.tolist()[:15]}...")
    
    # Find severity column - could be 'severity', 'severity_x', 'severity_y', or missing
    severity_col = None
    for col in ['severity', 'severity_x', 'severity_y']:
        if col in gdf.columns:
            severity_col = col
            break
    
    # Find other columns
    prob_col = None
    for col in ['ml_probability', 'ml_probability_x', 'ml_probability_y']:
        if col in gdf.columns:
            prob_col = col
            break
    
    burn_col = None
    for col in ['burn_percentage', 'burn_percentage_x', 'burn_percentage_y']:
        if col in gdf.columns:
            burn_col = col
            break
    
    dnbr_col = None
    for col in ['dnbr_mean', 'dnbr_mean_x', 'dnbr_mean_y']:
        if col in gdf.columns:
            dnbr_col = col
            break
    
    cluster_col = None
    for col in ['clustering_damaged', 'clustering_damaged_x', 'clustering_damaged_y']:
        if col in gdf.columns:
            cluster_col = col
            break
    
    ml_found_col = None
    for col in ['found_by_ml', 'found_by_ml_x', 'found_by_ml_y']:
        if col in gdf.columns:
            ml_found_col = col
            break
    
    # If severity is missing, use CSV data
    if severity_col is None:
        print("  No severity column found in GeoJSON, using CSV data...")
        
        # Find merge column
        merge_col = None
        for col in ['raster_id', 'building_id', 'id']:
            if col in df.columns and col in gdf.columns:
                merge_col = col
                break
        
        if merge_col:
            # Keep only needed columns from CSV
            csv_cols = [merge_col, 'severity', 'ml_probability', 'burn_percentage', 
                       'dnbr_mean', 'clustering_damaged', 'found_by_ml', 'n_pixels']
            csv_cols = [c for c in csv_cols if c in df.columns]
            df_subset = df[csv_cols].copy()
            
            # Merge
            gdf = gdf.merge(df_subset, on=merge_col, how='left')
            print(f"  Merged: {len(gdf):,} buildings")
            
            # Now use the merged columns
            severity_col = 'severity'
            prob_col = 'ml_probability'
            burn_col = 'burn_percentage'
            dnbr_col = 'dnbr_mean'
            cluster_col = 'clustering_damaged'
            ml_found_col = 'found_by_ml'
        else:
            # If can't merge, use default values
            print("  No merge column found, using defaults")
            gdf['severity'] = 3
            gdf['ml_probability'] = 0.8
            gdf['burn_percentage'] = 50
            gdf['dnbr_mean'] = 0.15
            gdf['clustering_damaged'] = 1
            gdf['found_by_ml'] = False
            severity_col = 'severity'
            prob_col = 'ml_probability'
            burn_col = 'burn_percentage'
            dnbr_col = 'dnbr_mean'
            cluster_col = 'clustering_damaged'
            ml_found_col = 'found_by_ml'
    
    # Now standardize column names
    if severity_col and severity_col != 'severity':
        gdf['severity'] = gdf[severity_col]
    elif 'severity' not in gdf.columns:
        gdf['severity'] = 3
    
    if prob_col and prob_col != 'ml_probability':
        gdf['ml_probability'] = gdf[prob_col]
    elif 'ml_probability' not in gdf.columns:
        gdf['ml_probability'] = 0.8
    
    if burn_col and burn_col != 'burn_percentage':
        gdf['burn_percentage'] = gdf[burn_col]
    elif 'burn_percentage' not in gdf.columns:
        gdf['burn_percentage'] = 50
    
    if dnbr_col and dnbr_col != 'dnbr_mean':
        gdf['dnbr_mean'] = gdf[dnbr_col]
    elif 'dnbr_mean' not in gdf.columns:
        gdf['dnbr_mean'] = 0.15
    
    if cluster_col and cluster_col != 'clustering_damaged':
        gdf['clustering_damaged'] = gdf[cluster_col]
    elif 'clustering_damaged' not in gdf.columns:
        gdf['clustering_damaged'] = 1
    
    if ml_found_col and ml_found_col != 'found_by_ml':
        gdf['found_by_ml'] = gdf[ml_found_col]
    elif 'found_by_ml' not in gdf.columns:
        gdf['found_by_ml'] = False
    
    # Ensure severity is int
    gdf['severity'] = gdf['severity'].fillna(3).astype(int)
    gdf['ml_probability'] = gdf['ml_probability'].fillna(0.8)
    gdf['burn_percentage'] = gdf['burn_percentage'].fillna(50)
    gdf['dnbr_mean'] = gdf['dnbr_mean'].fillna(0.15)
    gdf['clustering_damaged'] = gdf['clustering_damaged'].fillna(1).astype(int)
    gdf['found_by_ml'] = gdf['found_by_ml'].fillna(False).astype(bool)
    
    # Add severity label
    severity_map = {0: 'Undamaged', 1: 'Low', 2: 'Moderate', 3: 'High'}
    gdf['severity_label'] = gdf['severity'].map(severity_map).fillna('High')
    
    # Add discovery method
    gdf['discovery'] = 'Unknown'
    gdf.loc[(gdf['clustering_damaged'] == 1) & (gdf['found_by_ml'] == False), 'discovery'] = 'Clustering'
    gdf.loc[(gdf['clustering_damaged'] == 0) & (gdf['found_by_ml'] == True), 'discovery'] = 'ML'
    gdf.loc[(gdf['clustering_damaged'] == 1) & (gdf['found_by_ml'] == True), 'discovery'] = 'Both'
    
    print(f"\n  Final columns: {gdf.columns.tolist()[:10]}...")
    print(f"  Severity distribution: {gdf['severity'].value_counts().to_dict()}")
    print(f"  Discovery distribution: {gdf['discovery'].value_counts().to_dict()}")
    
    return gdf

# ============================================================================
# MAP CREATION - SIMPLIFIED
# ============================================================================

def create_map(gdf):
    """Create simplified interactive map"""
    
    print("\nCreating map...")
    
    # Base map - OpenStreetMap
    m = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=14,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Alternative basemaps
    # Satellite
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='ESRI',
        name='Satellite'
    ).add_to(m)
    
    # CartoDB dark (for contrast)
    folium.TileLayer(
        'CartoDB dark_matter',
        name='Dark'
    ).add_to(m)
    
    # CartoDB light
    folium.TileLayer(
        'CartoDB positron',
        name='Light'
    ).add_to(m)
    
    # ========================================================================
    # DAMAGE LAYER - Building footprints by severity
    # ========================================================================
    
    print("  Adding damage layer...")
    
    severity_colors = {
        3: {'color': '#FF0000', 'label': 'High Severity', 'opacity': 0.85},
        2: {'color': '#FF6B00', 'label': 'Moderate Severity', 'opacity': 0.75},
        1: {'color': '#FFD700', 'label': 'Low Severity', 'opacity': 0.65},
        0: {'color': '#00CC66', 'label': 'Undamaged', 'opacity': 0.35}
    }
    
    # Only include damaged buildings (severity 1-3)
    damaged_gdf = gdf[gdf['severity'] > 0].copy()
    print(f"  Showing {len(damaged_gdf):,} damaged buildings")
    
    if len(damaged_gdf) == 0:
        print("  No damaged buildings to display")
        return m
    
    # Add each severity layer
    for severity, config in severity_colors.items():
        subset = damaged_gdf[damaged_gdf['severity'] == severity]
        if len(subset) == 0:
            continue
        
        # Simplify geometry for performance
        try:
            subset['geometry'] = subset['geometry'].buffer(0).simplify(0.3)
        except:
            pass
        
        def style_function(feature, color=config['color'], opacity=config['opacity']):
            return {
                'fillColor': color,
                'color': 'white',
                'weight': 1,
                'fillOpacity': opacity,
            }
        
        # Tooltip fields
        tooltip_fields = ['severity_label']
        tooltip_aliases = ['Severity']
        
        if 'burn_percentage' in subset.columns:
            tooltip_fields.append('burn_percentage')
            tooltip_aliases.append('Burn %')
        if 'dnbr_mean' in subset.columns:
            tooltip_fields.append('dnbr_mean')
            tooltip_aliases.append('dNBR')
        if 'ml_probability' in subset.columns:
            tooltip_fields.append('ml_probability')
            tooltip_aliases.append('Confidence')
        if 'discovery' in subset.columns:
            tooltip_fields.append('discovery')
            tooltip_aliases.append('Found By')
        
        folium.GeoJson(
            subset,
            name=f"{config['label']} ({len(subset)})",
            style_function=style_function,
            highlight_function=lambda x: {'weight': 3, 'color': '#FFFFFF', 'fillOpacity': 0.95},
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                localize=True,
                style="""
                    background-color: rgba(20, 20, 30, 0.95);
                    border: 2px solid #FF6B00;
                    border-radius: 8px;
                    color: #FFFFFF;
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    padding: 8px;
                """
            )
        ).add_to(m)
    
    # ========================================================================
    # DISCOVERY METHOD LAYER (optional, can be toggled)
    # ========================================================================
    
    if 'discovery' in damaged_gdf.columns:
        print("  Adding discovery method layer...")
        
        discovery_group = folium.FeatureGroup(name="Discovery Method", show=False)
        
        # Clustering
        cluster_only = damaged_gdf[damaged_gdf['discovery'] == 'Clustering']
        if len(cluster_only) > 0:
            folium.GeoJson(
                cluster_only,
                name=f"Clustering ({len(cluster_only)})",
                style_function=lambda x: {'fillColor': '#3D7ED9', 'color': 'white', 'weight': 1, 'fillOpacity': 0.6},
                tooltip=folium.GeoJsonTooltip(
                    fields=['severity_label', 'discovery'],
                    aliases=['Severity', 'Method'],
                    localize=True
                )
            ).add_to(discovery_group)
        
        # ML
        ml_only = damaged_gdf[damaged_gdf['discovery'] == 'ML']
        if len(ml_only) > 0:
            folium.GeoJson(
                ml_only,
                name=f"ML ({len(ml_only)})",
                style_function=lambda x: {'fillColor': '#9B59B6', 'color': 'white', 'weight': 1, 'fillOpacity': 0.6},
                tooltip=folium.GeoJsonTooltip(
                    fields=['severity_label', 'discovery'],
                    aliases=['Severity', 'Method'],
                    localize=True
                )
            ).add_to(discovery_group)
        
        # Both
        both = damaged_gdf[damaged_gdf['discovery'] == 'Both']
        if len(both) > 0:
            folium.GeoJson(
                both,
                name=f"Both ({len(both)})",
                style_function=lambda x: {'fillColor': '#FF6B00', 'color': 'white', 'weight': 1, 'fillOpacity': 0.6},
                tooltip=folium.GeoJsonTooltip(
                    fields=['severity_label', 'discovery'],
                    aliases=['Severity', 'Method'],
                    localize=True
                )
            ).add_to(discovery_group)
        
        discovery_group.add_to(m)
    
    # ========================================================================
    # SUMMARY PANEL
    # ========================================================================
    
    print("  Adding summary panel...")
    
    total = len(gdf)
    damaged = len(damaged_gdf)
    
    sev_counts = damaged_gdf['severity'].value_counts().to_dict()
    sev_high = sev_counts.get(3, 0)
    sev_mod = sev_counts.get(2, 0)
    sev_low = sev_counts.get(1, 0)
    
    # Count by discovery method
    if 'discovery' in damaged_gdf.columns:
        cluster_count = len(damaged_gdf[damaged_gdf['discovery'] == 'Clustering'])
        ml_count = len(damaged_gdf[damaged_gdf['discovery'] == 'ML'])
        both_count = len(damaged_gdf[damaged_gdf['discovery'] == 'Both'])
    else:
        cluster_count = 0
        ml_count = 0
        both_count = 0
    
    stats_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000; 
                background: rgba(13, 13, 20, 0.95); 
                padding: 15px 20px; 
                border-radius: 12px; 
                border: 1px solid rgba(255, 107, 0, 0.3);
                box-shadow: 0 4px 20px rgba(0,0,0,0.9);
                color: #FFF;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-width: 200px;
                backdrop-filter: blur(10px);">
        <h4 style="margin: 0 0 10px 0; color: #FF6B00; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
            Eaton Fire Damage
        </h4>
        <div style="font-size: 13px;">
            <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                <span style="color: rgba(255,255,255,0.6);">Total Analyzed:</span>
                <span style="font-weight: bold;">{total:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0; color: #FF0000;">
                <span>Destroyed:</span>
                <span style="font-weight: bold;">{sev_high:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0; color: #FF6B00;">
                <span>Damaged:</span>
                <span style="font-weight: bold;">{damaged:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0; color: #FFD700;">
                <span>Low Damage:</span>
                <span style="font-weight: bold;">{sev_low:,}</span>
            </div>
            <hr style="margin: 8px 0; border-color: rgba(255,255,255,0.1);">
            <div style="display: flex; justify-content: space-between; margin: 5px 0; font-size: 11px;">
                <span style="color: rgba(255,255,255,0.5);">Clustering:</span>
                <span style="font-weight: bold; color: #3D7ED9;">{cluster_count:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0; font-size: 11px;">
                <span style="color: rgba(255,255,255,0.5);">ML Refinement:</span>
                <span style="font-weight: bold; color: #9B59B6;">{ml_count:,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 5px 0; font-size: 11px;">
                <span style="color: rgba(255,255,255,0.5);">Both:</span>
                <span style="font-weight: bold; color: #FF6B00;">{both_count:,}</span>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 9px; color: rgba(255,255,255,0.25); border-top: 1px solid rgba(255,255,255,0.05); padding-top: 5px;">
            Sentinel-2 10m | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(stats_html))
    
    # ========================================================================
    # LEGEND
    # ========================================================================
    
    legend_html = '''
    <div style="position: fixed; bottom: 20px; left: 50px; z-index: 1000; 
                background: rgba(13, 13, 20, 0.95); 
                padding: 12px 15px; 
                border-radius: 12px; 
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 20px rgba(0,0,0,0.9);
                color: #FFF;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 12px;
                backdrop-filter: blur(10px);">
        <h5 style="margin: 0 0 8px 0; color: #FF6B00;">Legend</h5>
        <div style="display: grid; grid-template-columns: 1fr; gap: 3px 15px;">
            <div><span style="color: #FF0000;">&#9679;</span> High Severity</div>
            <div><span style="color: #FF6B00;">&#9679;</span> Moderate Severity</div>
            <div><span style="color: #FFD700;">&#9679;</span> Low Severity</div>
        </div>
        <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 10px; color: rgba(255,255,255,0.3);">
            Hover for details | Toggle layers (top-right)
        </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # ========================================================================
    # CONTROLS
    # ========================================================================
    
    Fullscreen().add_to(m)
    MousePosition().add_to(m)
    MeasureControl().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    
    return m

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("EATON FIRE - DAMAGE MAP")
    print("Simplified version - Building footprints only")
    print("Basemap: OpenStreetMap")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    gdf = load_data()
    if gdf is None:
        print("Error: Could not load data")
        return
    
    if len(gdf) == 0:
        print("No data loaded")
        return
    
    # Create map
    m = create_map(gdf)
    
    # Save
    print(f"\nSaving map to: {OUTPUT_FILE}")
    m.save(OUTPUT_FILE)
    
    # Open in browser
    print("\nOpening in browser...")
    webbrowser.open(f'file://{os.path.abspath(OUTPUT_FILE)}')
    
    print("\n" + "="*60)
    print("MAP COMPLETE!")
    print("="*60)
    print(f"\nFile: {OUTPUT_FILE}")
    print("  Basemaps available (toggle top-right):")
    print("  - OpenStreetMap (default)")
    print("  - Satellite")
    print("  - Dark")
    print("  - Light")
    print("  Layers:")
    print("  - Damage Status (High/Moderate/Low) - Building footprints")
    print("  - Discovery Method (Clustering/ML/Both) - optional toggle")
    print("\n  Use layer control (top-right) to toggle layers")
    print("  Hover buildings for details")

if __name__ == "__main__":
    main()