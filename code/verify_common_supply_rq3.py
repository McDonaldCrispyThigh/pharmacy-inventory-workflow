"""Independent cached-input verification of the common-frame spatial outputs."""
from pathlib import Path
import json,socket
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pyproj import Transformer
import run_common_supply_rq3 as current
frozen=current.frozen
OUT,ROOT=current.OUT,current.ROOT
socket.socket.connect=frozen.old.deny_network;socket.create_connection=frozen.old.deny_network
read=frozen.read_json
save=current.save
frame=set(read(OUT/'common_supply_frame.json')['common_zips'])
candidates,_=frozen.load_candidates()
context=frozen.delivery.load_forced_choice_local_context(ROOT/'analysis/input_manifest.json',frozen.old.TARGET_ZIPS)
origins=context.origins.sort_values('GEOID').reset_index(drop=True)
tract=pd.read_csv(OUT/'rq3/tract_spatial.csv',dtype={'GEOID':str}).sort_values('GEOID').reset_index(drop=True)
prior=pd.read_csv(current.BASE/'rq3/tract_spatial.csv',dtype={'GEOID':str}).sort_values('GEOID').reset_index(drop=True)
assert origins.GEOID.tolist()==tract.GEOID.tolist()==prior.GEOID.tolist()
for col in ['target_zcta','COUNTYFP','med_hh_income','total_pop','pct_nonwhite','total_hh','carless_rate','income_quartile','nonwhite_quartile']:
    assert tract[col].equals(prior[col]),col
google,boards=current.common_engine(frame,len(candidates))._google_and_boards(candidates,context,frozen.old.TARGET_ZIPS)
surfaces={
 'google_include':(google.loc[google.eligibility_state.eq('include')],'Longitude','Latitude'),
 'google_include_unresolved':(google.loc[google.eligibility_state.isin(['include','unresolved'])],'Longitude','Latitude'),
 'board_screened':(boards.screened_extended,'_lon','_lat'),
 'board_screened_exact':(boards.screened_extended.loc[boards.screened_extended.mtype.eq('Exact')],'_lon','_lat'),
 'board_all_active':(boards.full_active_extended,'_lon','_lat')}
xy=np.column_stack([origins.geometry.x,origins.geometry.y])
assert origins.crs.to_epsg()==26915
transform=Transformer.from_crs(4326,26915,always_xy=True)
checks={};facility_xy={}
for name,(data,lon,lat) in surfaces.items():
    x=pd.to_numeric(data[lon],errors='coerce');y=pd.to_numeric(data[lat],errors='coerce')
    valid=np.isfinite(x)&np.isfinite(y)&x.between(-180,180)&y.between(-90,90)
    px,py=transform.transform(x.loc[valid].to_numpy(),y.loc[valid].to_numpy())
    facility_xy[name]=np.column_stack([px,py])
    distance=cdist(xy,facility_xy[name]).min(axis=1)
    error=float(abs(distance-tract[name+'_nearest_m'].to_numpy()).max())
    assert error<1e-6,(name,error)
    checks[name]={'records':len(data),'usable_coordinates':int(valid.sum()),'max_independent_error_m':error}
zctas=gpd.read_file(f"zip://{(ROOT/'data/tl_2023_us_zcta520.zip').as_posix()}")
zctas['zip5']=zctas.ZCTA5CE20.astype(str).str.zfill(5)
polygons=zctas.loc[zctas.zip5.isin(frame)].to_crs(26915)
assert not polygons.zip5.duplicated().any()
assert polygons.geometry.notna().all() and not polygons.geometry.is_empty.any() and polygons.geometry.is_valid.all()
union=polygons.geometry.union_all()
assert union.is_valid and all(union.covers(p) for p in origins.geometry)
missing=sorted(frame-set(polygons.zip5))
edge=origins.geometry.distance(union.boundary).to_numpy()
board_missing=boards.screened_extended.loc[~boards.screened_extended.has_geocode]
missingzip=set(board_missing.zip5)
mailpolygons=zctas.loc[zctas.zip5.isin(missingzip)].to_crs(26915)
unmapped=sorted(missingzip-set(mailpolygons.zip5))
mindistance=origins.geometry.distance(mailpolygons.geometry.union_all()).to_numpy()
if unmapped:mindistance=np.zeros(len(origins))
tractpolygons=gpd.read_file(f"zip://{(ROOT/'data/tl_2023_27_tract.zip').as_posix()}")
tractpolygons=tractpolygons.loc[tractpolygons.GEOID.isin(origins.GEOID)].to_crs(26915).sort_values('GEOID').reset_index(drop=True)
centroids=tractpolygons.geometry.centroid
centroid_xy=np.column_stack([centroids.x,centroids.y])
cg=cdist(centroid_xy,facility_xy['google_include']).min(axis=1)
cb=cdist(centroid_xy,facility_xy['board_screened']).min(axis=1)
extension=read(OUT/'rq3/spatial_sensitivity_extension.json')
assert missing==extension['collection_frame']['missing_zcta_geometry']
assert unmapped==extension['missing_board_coordinates']['mailing_zctas_without_geometry']
def paired(w,b):
    return {'paired_origins':len(w),'both_within':int((~w&~b).sum()),'both_beyond':int((w&b).sum()),'workflow_only_within':int((~w&b).sum()),'board_only_within':int((w&~b).sum()),'classification_changed_count':int((w!=b).sum()),'classification_changed_pct':round(100*float((w!=b).mean()),2)}
details=pd.DataFrame({'GEOID':origins.GEOID,'collection_edge_m':edge,'missing_board_minimum_possible_m':mindistance,'workflow_centroid_nearest_m':cg,'board_centroid_nearest_m':cb})
for threshold,extra in extension['thresholds'].items():
    radius=float(threshold)*1609.344
    w=tract.google_include_nearest_m.to_numpy()>radius;b=tract.board_screened_nearest_m.to_numpy()>radius
    interior=edge>radius
    assert paired(w[interior],b[interior])==extra['collection_frame_interior']['comparison']
    assert paired(cg>radius,cb>radius)==extra['geometric_centroid_origin']
    wd=w&interior;bd=b&interior&(mindistance>radius)
    wu=w&~wd;bu=b&~bd;determinate=~(wu|bu)
    expected=extra['conditional_ranges']
    assert paired(w[determinate],b[determinate])==expected['conditionally_determinate_comparison']
    for name,definite,uncertain in [('workflow',wd,wu),('board',bd,bu)]:
        assert round(100*float(definite.mean()),2)==expected[name+'_beyond_rate_lower_pct']
        assert round(100*float((definite|uncertain).mean()),2)==expected[name+'_beyond_rate_upper_pct']
        assert int(uncertain.sum())==expected[name+'_conditionally_unresolved_count']
    details['interior_'+threshold]=interior
    details['workflow_conditional_'+threshold]=np.where(~w,'within',np.where(wd,'beyond','unresolved'))
    details['board_conditional_'+threshold]=np.where(~b,'within',np.where(bd,'beyond','unresolved'))
details.to_csv(OUT/'tract_geometry_sensitivity_ledger.csv',index=False)
save(OUT/'geometry_audit.json',{'common_postal_zip_count':len(frame),'represented_zcta_count':len(polygons),'missing_zcta_geometry':missing,'invalid_or_empty_geometries':0,'union_valid':bool(union.is_valid),'origin_points_covered':len(origins),'distance_crs':'EPSG:26915','union_area_square_meters':float(union.area),'union_wkb_sha256':frozen.digest(union.wkb),'geometry_source_sha256':frozen.sha(ROOT/'data/tl_2023_us_zcta520.zip'),'missing_board_zip_without_geometry':unmapped,'missing_geometry_rule_applied':bool(unmapped)})
save(OUT/'independent_verification.json',{'passed':True,'method':'pyproj Transformer and SciPy cdist, independently checked against five delivered surfaces; direct polygon boundary and centroid reconstruction for all predefined geometric conditions.','surface_checks':checks,'unchanged_origins_demographics_quartiles':True,'all_centroid_interior_conditional_checks_passed':True})
print(json.dumps({'passed':True,'surface_counts':checks,'represented_zctas':len(polygons),'missing_geometry':missing}))
