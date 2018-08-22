#ifndef MOVIES_H
#define MOVIES_H

#include "Movie.h"
#include <iostream>
#include <string>
#include <vector>

class Movies {
private:
	std::vector <Movie> movies;
	
	bool movie_exists(std::string name) {
		for(Movie &movie : this->movies) {
			if (name == movie.get_title()) {
				return true;
			}
		}
		return false;
	}
	
	Movie *get_movie(std::string name) {
		for(Movie &movie : movies) {
			if (movie.get_title() == name) {
				return &movie;
			}
		}
		return nullptr;
	}
	
	
public:
	Movies();
	~Movies();
	
	bool add_movie(std::string title, std::string rating, int watched = 0);
	bool watch_movie(std::string title);
	void display() const;

};

#endif // MOVIES_H
