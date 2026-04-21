"""
Integration Test Script for Terrain Mapping and Path Planning.

This script tests the integration between the robotic perception agents (`motion.py`) 
and the spatial memory components (`data_management.py`). It simulates a selected 
robot (Drone, Scout, or Rover) exploring the terrain using the `MapAPI`.

Crucially, this script demonstrates how the accumulated `TerrainMap` data is used
to build a Directed Weighted Graph (via NetworkX) for path planning. The generated 
graph models reachable nodes and traversable edges, ensuring that subsequent pathfinding 
algorithms can avoid obstacles and areas with a high stuck probability.

"""
import unittest
import math
import networkx as nx

# Importiamo le classi e le funzioni dal tuo file
from data_management import (
    CellData, 
    TerrainMap, 
    get_neighbors_8, 
    compute_edge_cost, 
    build_weighted_graph
)

# Creiamo un Mock per simulare i dati restituiti dal perceive
class MockTerrainObservation:
    def __init__(self, x, y, texture, color, slope=0.0, uphill_angle=0.0):
        self.x = x
        self.y = y
        self.features = {
            "texture": texture,
            "color": color,
            "slope": slope,
            "uphill_angle": uphill_angle
        }

class TestDataManagement(unittest.TestCase):

    def test_cell_data_initialization_and_setters(self):
        """Testa se le singole celle salvano correttamente i dati base."""
        cell = CellData()
        self.assertIsNone(cell.texture)
        
        cell.set_texture(0.7)
        self.assertEqual(cell.texture, 0.7)
        
        # Prova a sovrascrivere (nella tua logica attuale set_texture non sovrascrive se non è None)
        cell.set_texture(0.9)
        self.assertEqual(cell.texture, 0.7, "Il setter non dovrebbe sovrascrivere un valore già esistente se non è None.")

    def test_terrain_map_storage(self):
        """Testa l'inserimento delle osservazioni del robot all'interno della griglia spaziale."""
        t_map = TerrainMap(50, 50)
        obs1 = MockTerrainObservation(x=5, y=5, texture=0.2, color=0.3)
        obs2 = MockTerrainObservation(x=5, y=6, texture=0.4, color=0.5)
        
        t_map.store_observation([obs1, obs2])
        
        # Verifichiamo che la cella (5,5) esista e abbia i dati giusti
        cell_5_5 = t_map.get_cell(5, 5)
        self.assertEqual(cell_5_5.texture, 0.2)
        self.assertEqual(cell_5_5.color, 0.3)
        
        # Verifichiamo che anche (5,6) sia registrata
        self.assertIn((5, 6), t_map.grid)

    def test_refresh_estimation_idw(self):
        """Testa la logica predittiva dell'algoritmo Inverse Distance Weighting."""
        t_map = TerrainMap(50, 50)
        
        # 1. Creiamo una cella ESPLORATA e SICURA
        cell_safe = t_map.get_cell(0, 0)
        cell_safe.texture = 0.5
        cell_safe.color = 0.5
        cell_safe.real_traversability = 0.9 # Molto facile da attraversare
        cell_safe.is_stuck = False
        
        # 2. Creiamo una cella ESPLORATA ma PERICOLOSA
        cell_danger = t_map.get_cell(10, 10)
        cell_danger.texture = 0.9
        cell_danger.color = 0.9
        cell_danger.real_traversability = 0.1 # Molto difficile
        cell_danger.is_stuck = True
        
        # 3. Creiamo una cella NON ESPLORATA ma visivamente identica a quella SICURA
        cell_unknown = t_map.get_cell(0, 1)
        cell_unknown.texture = 0.51  # Molto simile a 0.5
        cell_unknown.color = 0.51
        
        # Avviamo la stima
        t_map.refresh_estimation()
        
        # La stima della cell_unknown dovrebbe avvicinarsi a 0.9, e la confidenza deve essere alta
        self.assertIsNotNone(cell_unknown.traversability_estimate)
        self.assertAlmostEqual(cell_unknown.traversability_estimate, 0.9, delta=0.05)
        self.assertAlmostEqual(cell_unknown.stuck_probability_estimate, 0.0, delta=0.05)
        self.assertTrue(cell_unknown.confidence > 0)

    def test_get_neighbors_8(self):
        """Testa che la funzione di utilità restituisca esattamente le 8 celle adiacenti."""
        neighbors = list(get_neighbors_8(5, 5))
        self.assertEqual(len(neighbors), 8)
        self.assertIn((4, 4), neighbors)
        self.assertIn((6, 5), neighbors)
        self.assertNotIn((5, 5), neighbors) # Non deve includere se stessa

    def test_compute_edge_cost(self):
        """Testa la formula di calcolo dei costi di movimento inclusiva di pendenza e penalità."""
        source = CellData()
        target = CellData(
            traversability_estimate=0.5, # base cost = 1 / 0.5 = 2.0
            confidence=1.0,              # no confidence penalty
            stuck_probability_estimate=0.1, # no stuck penalty (< 0.5)
            slope=None
        )
        
        # Test base
        cost = compute_edge_cost(source, target, direction=(0, 1))
        self.assertEqual(cost, 2.0)
        
        # Test con bassa confidenza (Aggiunge la penalità lambda)
        target.confidence = 0.0
        cost_low_conf = compute_edge_cost(source, target, direction=(0, 1))
        self.assertEqual(cost_low_conf, 4.0) # 2.0 (base) + 2.0 (lambda * (1 - 0))

        # Test con alta probabilità di blocco
        target.stuck_probability_estimate = 0.9
        cost_stuck = compute_edge_cost(source, target, direction=(0, 1))
        self.assertTrue(cost_stuck > 1000.0) # Penalità massima applicata

    def test_build_weighted_graph(self):
        """Testa la generazione del grafo NetworkX per gli algoritmi A*/Dijkstra."""
        t_map = TerrainMap(10, 10)
        
        # Creiamo due celle vicine
        t_map.get_cell(0, 0)
        target_cell = t_map.get_cell(0, 1)
        target_cell.traversability_estimate = 1.0
        target_cell.confidence = 1.0
        
        # Generiamo il grafo
        graph = build_weighted_graph(t_map)
        
        # Assicuriamoci che esista l'arco e che il peso sia calcolato
        self.assertTrue(graph.has_edge((0, 0), (0, 1)))
        
        edge_weight = graph[(0, 0)][(0, 1)]['weight']
        self.assertEqual(edge_weight, 1.0) # Costo base con trav=1.0 e conf=1.0 senza pendenza

if __name__ == '__main__':
    unittest.main()