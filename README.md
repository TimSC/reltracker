reltracker
==========

Implementation of: Tim Sheerman-Chase, Eng-Jon Ong, Richard Bowden. Non-linear Predictors 
for Facial feature Tracking across Pose and Expression. In IEEE 
Conference on Automatic Face and Gesture Recognition, Shanghai, 2013.

This library depends on PIL, numpy and sklearn.

reltracker.py contains the main class RelTracker. This version depends on cython 
optimisations which can be compiled by:

 python setup.py build_ext --inplace

reltrackersimple.py is a simplified implementation without speed optimisation and 
less flexibility. No cython is required.

legal
=====

Copyright (C) 2013 Tim Sheerman-Chase

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
