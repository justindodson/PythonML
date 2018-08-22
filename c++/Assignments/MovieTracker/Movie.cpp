#include "Movie.h"


Movie::Movie(std::string m_title, std::string m_rating, int m_times_watched) 
	: title (m_title), rating (m_rating), times_watched (m_times_watched) {
}

Movie::Movie(const Movie &source) 
	: Movie{source.title, source.rating, source.times_watched} {
}


Movie::~Movie() {
}

void Movie::display() const {
		std::cout << title << ": " << rating << " watched: " << times_watched << " times";
	}


