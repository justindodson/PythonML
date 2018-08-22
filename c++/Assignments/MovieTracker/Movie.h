#ifndef MOVIE_H
#define MOVIE_H

#include <string>
#include <iostream>

class Movie {
private:
	std::string title;
	std::string rating; // TODO: Create enum for movie rating that is used for rating options
	int times_watched;
	
public:
	Movie(std::string m_title, std::string m_rating, int m_times_watched);
	Movie(const Movie &source);
	~Movie();
	
	// methods
	void set_title(std::string title) {
		this->title = title;
	}
	std::string get_title() const {
		return title;
	}
	
	void set_rating(std::string rating) {
		this->rating = rating;
	}
	std::string get_rating() const {
		return rating;
	}
	
	void set_times_watched(int watched) {
		this->times_watched = watched;
	}
	int get_times_watched() const {
		return times_watched;
	}
	
	void watch() { ++times_watched; }
	
	void display() const;
	
	
};

#endif // MOVIE_H
