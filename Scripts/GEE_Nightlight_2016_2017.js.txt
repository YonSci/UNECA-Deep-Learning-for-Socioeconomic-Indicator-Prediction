// Define the region of interest
var region = ee.Geometry.Rectangle([31.0, -18, 38, -8.5]);

// Define the start and end dates for the desired time range
var startDate = '2016-04-01';
var endDate = '2017-04-30';

// Load the night lights dataset
var nightLights = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG')
                  .filterDate(startDate, endDate)
                  .select('avg_rad')
                  .filterBounds(region)
                  .mean();

// Clip the nightLights to the region of interest
var clippednightLights = nightLights.clip(region);

// Display the night lights on the map
Map.centerObject(region, 5);
Map.addLayer(clippednightLights, {min: 0, max: 100}, 'Night Lights');


// Export the night lights image as a TIFF file to Google Drive
Export.image.toDrive({
  image: nightLights,
  description: 'NightLights_2016_2017',
  folder: 'NightLights',
  region: region,
  scale: 30,  // Adjust the scale if needed
  crs: 'EPSG:4326',  // Adjust the CRS if needed
  maxPixels: 1e13  // Set a higher maxPixels value
});
