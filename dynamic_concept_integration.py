#!/usr/bin/env python3
"""
Dynamic Concept Integration for Ontology Completion
This script shows how to integrate new concepts into existing ontologies
using our trained unary classification model.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F

class DynamicConceptIntegrator:
    def __init__(self, trained_model, existing_graph, existing_concepts):
        """
        Initialize the dynamic concept integrator
        
        Args:
            trained_model: Pre-trained RGCN model
            existing_graph: Current ontology graph
            existing_concepts: List of existing concept names
        """
        self.model = trained_model
        self.graph = existing_graph
        self.existing_concepts = existing_concepts
        self.concept_embeddings = {}
        
    def create_concept_embedding(self, concept_name, embedding_dim=32):
        """
        Create embedding for a new concept
        
        Args:
            concept_name: Name of the new concept
            embedding_dim: Dimension of the embedding
            
        Returns:
            torch.Tensor: Embedding vector
        """
        # For demonstration, create a random embedding
        # In practice, you would use word embeddings, ontology embeddings, etc.
        embedding = torch.randn(embedding_dim)
        
        # Normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=0)
        
        return embedding
    
    def predict_relationships(self, new_concept, confidence_threshold=0.5):
        """
        Predict relationships between new concept and existing concepts
        
        Args:
            new_concept: Name of the new concept
            confidence_threshold: Minimum confidence for accepting a relationship
            
        Returns:
            list: List of predicted relationships
        """
        # Create embedding for new concept
        new_embedding = self.create_concept_embedding(new_concept)
        
        # Store the embedding
        self.concept_embeddings[new_concept] = new_embedding
        
        # Predict relationships with existing concepts
        predicted_relationships = []
        
        for existing_concept in self.existing_concepts:
            # Get existing concept embedding
            existing_embedding = self.concept_embeddings.get(existing_concept)
            if existing_embedding is None:
                # Create embedding for existing concept if not available
                existing_embedding = self.create_concept_embedding(existing_concept)
                self.concept_embeddings[existing_concept] = existing_embedding
            
            # Create a simple graph for prediction
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            edge_type = torch.tensor([0, 0], dtype=torch.long)
            
            # Combine embeddings
            combined_features = torch.stack([new_embedding, existing_embedding])
            
            # Create data object
            data = Data(
                x=combined_features,
                edge_index=edge_index,
                edge_type=edge_type
            )
            
            # Make prediction
            with torch.no_grad():
                self.model.eval()
                prediction = self.model(data)
                confidence = torch.sigmoid(prediction[0]).item()
            
            # Check if relationship is predicted
            if confidence > confidence_threshold:
                predicted_relationships.append({
                    'relationship': f"{new_concept} ⊑ {existing_concept}",
                    'confidence': confidence,
                    'type': 'subclass_of'
                })
        
        return predicted_relationships
    
    def integrate_concept(self, new_concept, confidence_threshold=0.5):
        """
        Integrate a new concept into the existing ontology
        
        Args:
            new_concept: Name of the new concept
            confidence_threshold: Minimum confidence for accepting relationships
            
        Returns:
            dict: Integration results
        """
        print(f"Integrating new concept: {new_concept}")
        print("=" * 50)
        
        # Predict relationships
        relationships = self.predict_relationships(new_concept, confidence_threshold)
        
        # Display results
        print(f"Predicted relationships for {new_concept}:")
        for rel in relationships:
            print(f"  ✅ {rel['relationship']} (confidence: {rel['confidence']:.3f})")
        
        # Update ontology
        self.existing_concepts.append(new_concept)
        
        # Create updated graph
        updated_graph = self.update_graph(new_concept, relationships)
        
        return {
            'new_concept': new_concept,
            'relationships': relationships,
            'updated_graph': updated_graph,
            'total_concepts': len(self.existing_concepts)
        }
    
    def update_graph(self, new_concept, relationships):
        """
        Update the graph structure with new concept and relationships
        
        Args:
            new_concept: Name of the new concept
            relationships: List of predicted relationships
            
        Returns:
            Data: Updated graph
        """
        # This is a simplified version
        # In practice, you would update the actual graph structure
        print(f"Graph updated with {new_concept} and {len(relationships)} new relationships")
        return self.graph
    
    def batch_integrate(self, new_concepts, confidence_threshold=0.5):
        """
        Integrate multiple new concepts at once
        
        Args:
            new_concepts: List of new concept names
            confidence_threshold: Minimum confidence for accepting relationships
            
        Returns:
            dict: Batch integration results
        """
        results = {}
        
        for concept in new_concepts:
            print(f"\nProcessing concept: {concept}")
            results[concept] = self.integrate_concept(concept, confidence_threshold)
        
        return results

def demo_dynamic_integration():
    """
    Demonstration of dynamic concept integration
    """
    print("🚀 Dynamic Concept Integration Demo")
    print("=" * 50)
    
    # Simulate existing ontology
    existing_concepts = [
        "Vehicle", "Car", "Truck", "Bicycle", "Motorcycle",
        "Road", "Highway", "Street", "Bridge",
        "Driver", "Passenger", "Pedestrian",
        "Machine", "Transport", "Infrastructure"
    ]
    
    # Create a simple graph (in practice, this would be your actual ontology graph)
    num_nodes = len(existing_concepts)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_type = torch.tensor([0, 0], dtype=torch.long)
    x = torch.randn(num_nodes, 32)
    
    graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    # Create a simple model (in practice, this would be your trained model)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = RGCNConv(32, 1, num_relations=1)
        
        def forward(self, data):
            return self.conv(data.x, data.edge_index, data.edge_type)
    
    model = SimpleModel()
    
    # Initialize integrator
    integrator = DynamicConceptIntegrator(model, graph, existing_concepts)
    
    # New concepts to integrate
    new_concepts = [
        "ElectricVehicle",
        "AutonomousVehicle", 
        "FlyingCar",
        "Hyperloop",
        "ElectricScooter"
    ]
    
    # Integrate concepts
    results = integrator.batch_integrate(new_concepts, confidence_threshold=0.6)
    
    # Display summary
    print("\n" + "=" * 50)
    print("INTEGRATION SUMMARY")
    print("=" * 50)
    
    for concept, result in results.items():
        print(f"\n{concept}:")
        print(f"  Relationships found: {len(result['relationships'])}")
        for rel in result['relationships']:
            print(f"    - {rel['relationship']} ({rel['confidence']:.3f})")
    
    return results

if __name__ == "__main__":
    # Run the demo
    results = demo_dynamic_integration()
    
    print("\n🎯 Dynamic concept integration completed!")
    print("New concepts have been successfully integrated into the ontology.")
