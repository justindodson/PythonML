#include <iostream>
#include <vector>
#include <string>
#include "Movies.h"

using namespace std;

int main() {
	Movies my_movie;
	
	my_movie.display();
	
	my_movie.add_movie("Twister", "PG-13", 0);
	my_movie.add_movie("Twister", "PG-13", 0); // tests for making sure you can't add same movie title

	my_movie.display();
	my_movie.watch_movie("Twister");
	my_movie.add_movie("The Matrix", "R", 0);
	my_movie.add_movie("Lord of the Rings", "PG-13", 0);
	
	my_movie.display();
	my_movie.watch_movie("The Matrix");
	my_movie.watch_movie("Lord of the Rings");
	my_movie.watch_movie("Twister");
	
	
	my_movie.watch_movie("Star Wars"); // test for watching a movie that doesn't exist
	my_movie.add_movie("Star Wars", "PG-13", 3);
	
	my_movie.watch_movie("Lord of the Rings");
	my_movie.watch_movie("The Matrix");
	my_movie.watch_movie("Lord of the Rings");
	my_movie.watch_movie("Star Wars");
	
	my_movie.display();
	
	
	
	
	return 0;
}