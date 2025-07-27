import overpy

def find_nearest_hospitals(lat, lon):
    try:
        api = overpy.Overpass()
        query = f"""
        [out:json];
        (
          node["amenity"="hospital"](around:5000,{lat},{lon});
        );
        out body;
        """
        result = api.query(query)

        if not result.nodes:
            return "No hospitals found nearby."

        hospital_names = [node.tags.get("name", "Unnamed Hospital") for node in result.nodes]
        return "\n".join(hospital_names[:5])  # Limit to top 5

    except Exception as e:
        return f"Error fetching hospitals: {str(e)}"
