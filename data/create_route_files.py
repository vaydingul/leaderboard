import os
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
def create_route_files(xml_file):
	# Create a directory to store the route files
	directory = os.path.dirname(os.path.abspath(xml_file))
	filename = os.path.basename(xml_file)

	# Create a folder called filename without the extension
	directory = os.path.join(directory, os.path.splitext(filename)[0])
	os.makedirs(directory, exist_ok=True)

	# Parse the XML file
	tree = ET.parse(xml_file)
	root = tree.getroot()

	# Iterate over each route element
	for route in root.findall('route'):
		# Get the route ID and town attributes
		route_id = route.get('id')
		town = route.get('town')

		# Create a new XML tree for the route
		route_tree = ET.ElementTree(route)
		
		

		# Create a separate XML file for each route
		route_file = os.path.join(directory, f"{town}_route{route_id}.xml")
		route_tree.write(route_file, encoding='utf-8', xml_declaration=True)

		print(f"Created file: {route_file}")



if __name__ == "__main__":

	argparser = ArgumentParser()
	argparser.add_argument(
		'-x', '--xml_file',
		default='leaderboard/data/routes_testing.xml',
		help='Path to the XML file containing the routes'
	)

	args = argparser.parse_args()
	create_route_files(args.xml_file)
	