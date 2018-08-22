#include "Movies.h"

Movies::Movies(){}

Movies::~Movies(){}

bool Movies::add_movie(std::string title, std::string rating, int watched) {
	if(movie_exists(title)) {
		std::cout << title << " already exists." << std::endl;
		return false;
	} else {
		movies.push_back(Movie(title, rating, watched)); 	
		std::cout << title << " added to movie collection " << std::endl;
		return true;
	}
}

bool Movies::watch_movie (std::string name) {
	if (movie_exists(name)) {
		get_movie(name)->watch();
		std::cout << "watched " << name << std::endl;
		return true;
	}else {
		std::cout << name << " does not exist in your movie files" << std::endl;
		return false;
	}
}

void Movies::display() const {
	if(movies.size() == 0) {
		std::cout << "Sorry, no movies in your collection to display." << std::endl;
	} else {
		std::cout << "\n======================================" << std::endl;
		for(Movie movie : movies) {
			movie.display();
			std::cout << std::endl;;
		}
		std::cout << "======================================" << std::endl << std::endl;
	}
	
}

