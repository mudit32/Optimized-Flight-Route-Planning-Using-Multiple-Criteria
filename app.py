import pandas as pd
import networkx as nx
import folium
import streamlit as st
from streamlit_folium import folium_static

class FlightOptimizer:
    def __init__(self, flight_data_path):
        self.graph = nx.DiGraph()
        self.flights = pd.read_csv(flight_data_path)
        self._build_graph()

    def _build_graph(self):
        for _, row in self.flights.iterrows():
            self.graph.add_edge(
                row['source'], row['dest'],
                cost=row['cost'],
                time=row['time_minutes'],
                co2=row['co2_kg'],
                source_lat=row['source_lat'],
                source_lon=row['source_lon'],
                dest_lat=row['dest_lat'],
                dest_lon=row['dest_lon']
            )

    def preprocess_combined_weights(self, weights):
        cost_w, time_w, layover_w, co2_w = weights
        for u, v, data in self.graph.edges(data=True):
            data['combined_score'] = (
                cost_w * data['cost'] +
                time_w * data['time'] +
                co2_w * data['co2'] +
                layover_w * 1
            )

    def score_path(self, path, weights):
        cost_w, time_w, layover_w, co2_w = weights
        total_cost, total_time, total_co2 = 0, 0, 0
        for i in range(len(path) - 1):
            edge = self.graph[path[i]][path[i + 1]]
            total_cost += edge['cost']
            total_time += edge['time']
            total_co2 += edge['co2']
        score = (
            total_cost * cost_w +
            total_time * time_w +
            total_co2 * co2_w +
            (len(path) - 2) * layover_w
        )
        return score, total_cost, total_time, total_co2, len(path) - 2

    def get_dijkstra_path(self, src, dst):
        path = nx.dijkstra_path(self.graph, src, dst, weight='combined_score')
        return path, self.score_path(path, self.current_weights)

    def get_bellman_ford_path(self, src, dst):
        path = nx.bellman_ford_path(self.graph, src, dst, weight='combined_score')
        return path, self.score_path(path, self.current_weights)

    def visualize_paths(self, paths_info):
        if not paths_info:
            return None

        color_map = {
            "Best": "red",
            "Alternative": "blue",
            "Dijkstra": "green",
            "Bellman-Ford": "orange"
        }

        lat = self.graph[paths_info[0]['path'][0]][paths_info[0]['path'][1]]['source_lat']
        lon = self.graph[paths_info[0]['path'][0]][paths_info[0]['path'][1]]['source_lon']
        fmap = folium.Map(location=[lat, lon], zoom_start=4)

        for info in paths_info:
            label = info['label']
            color = color_map.get(label, "gray")
            path = info['path']

            for i in range(len(path) - 1):
                edge = self.graph[path[i]][path[i + 1]]
                points = [(edge['source_lat'], edge['source_lon']),
                          (edge['dest_lat'], edge['dest_lon'])]
                tooltip = f"{path[i]} → {path[i+1]}<br>₹{edge['cost']}, {edge['time']} min, {edge['co2']} kg CO₂"
                folium.PolyLine(
                    points,
                    color=color,
                    weight=5 if label == "Best" else 3,
                    opacity=0.7,
                    tooltip=tooltip
                ).add_to(fmap)

            last_lat = self.graph[path[-2]][path[-1]]['dest_lat']
            last_lon = self.graph[path[-2]][path[-1]]['dest_lon']
            folium.Marker(
                [last_lat, last_lon],
                popup=(f"{label} Path<br>Score: {info['score']:.2f}<br>₹{info['cost']}, {info['time']} min, {info['co2']} kg CO₂, Layovers: {info['layovers']}"),
                icon=folium.Icon(color=color, icon='plane')
            ).add_to(fmap)

        return fmap

def main():
    st.title("Flight Route Optimizer (Multi-Criteria)")
    flight_data_path = "flights.csv"
    optimizer = FlightOptimizer(flight_data_path)

    airports = sorted(set(optimizer.flights['source']).union(optimizer.flights['dest']))
    src = st.selectbox("Select Source Airport", airports)
    dst = st.selectbox("Select Destination Airport", airports)

    st.markdown("### Rate your preferences (1 = least important, 10 = most important):")
    cost_w = st.slider("Cost Weight", 1, 10, 5)
    time_w = st.slider("Time Weight", 1, 10, 5)
    layover_w = st.slider("Layover Weight", 1, 10, 5)
    co2_w = st.slider("CO₂ Emission Weight", 1, 10, 5)
    weights = (cost_w, time_w, layover_w, co2_w)
    optimizer.current_weights = weights

    optimizer.preprocess_combined_weights(weights)
    all_paths_info = []

    if st.button("Find Routes"):
        try:
            d_path, d_score = optimizer.get_dijkstra_path(src, dst)
            st.success("Dijkstra Path: " + " -> ".join(d_path))
            st.write(f"Score: {d_score[0]:.2f}, Cost: ₹{d_score[1]}, Time: {d_score[2]} min, CO₂: {d_score[3]} kg, Layovers: {d_score[4]}")
            all_paths_info.append({
                "label": "Dijkstra", "path": d_path, "score": d_score[0], "cost": d_score[1],
                "time": d_score[2], "co2": d_score[3], "layovers": d_score[4]
            })

            b_path, b_score = optimizer.get_bellman_ford_path(src, dst)
            st.success("Bellman-Ford Path: " + " -> ".join(b_path))
            st.write(f"Score: {b_score[0]:.2f}, Cost: ₹{b_score[1]}, Time: {b_score[2]} min, CO₂: {b_score[3]} kg, Layovers: {b_score[4]}")
            all_paths_info.append({
                "label": "Bellman-Ford", "path": b_path, "score": b_score[0], "cost": b_score[1],
                "time": b_score[2], "co2": b_score[3], "layovers": b_score[4]
            })

            paths_gen = nx.shortest_simple_paths(optimizer.graph, src, dst, weight='combined_score')
            for i, path in enumerate(paths_gen):
                if i >= 5:
                    break
                score_data = optimizer.score_path(path, weights)
                label = "Best" if i == 0 else "Alternative"
                st.info(f"{label} Path {i+1}: {' -> '.join(path)}")
                st.write(f"Score: {score_data[0]:.2f}, Cost: ₹{score_data[1]}, Time: {score_data[2]} min, CO₂: {score_data[3]} kg, Layovers: {score_data[4]}")
                all_paths_info.append({
                    "label": label, "path": path, "score": score_data[0], "cost": score_data[1],
                    "time": score_data[2], "co2": score_data[3], "layovers": score_data[4]
                })

            fmap = optimizer.visualize_paths(all_paths_info)
            if fmap:
                folium_static(fmap)

        except nx.NetworkXNoPath:
            st.error("No valid path found between selected airports.")

if __name__ == "__main__":
    main()
