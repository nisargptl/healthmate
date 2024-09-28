from neo4j import GraphDatabase
class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_entity(self, entity_name, entity_type):
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name, type: $type})
            """, name=entity_name, type=entity_type)

    def create_relationship(self, entity1, relationship, entity2):
        with self.driver.session() as session:
            session.run("""
                MATCH (e1:Entity {name: $entity1})
                MATCH (e2:Entity {name: $entity2})
                MERGE (e1)-[:RELATIONSHIP {type: $relationship}]->(e2)
            """, entity1=entity1, entity2=entity2, relationship=relationship)

    def store_entities_and_relationships(self, entities, relationships):
        if(len(entities) == 0 and len(relationships) == 0):
            return
        for entity in entities:
            self.create_entity(entity['name'], entity['type'])
        for relationship in relationships:
            self.create_relationship(relationship['from'], relationship['relationship'], relationship['to'])


    def fetch_entities_and_relationships_for_user(self, user_name):
        with self.driver.session() as session:
            result = session.run("""
              MATCH (u:Entity {name: $user_name, type: "Person"})-[r]->(e:Entity)
              RETURN DISTINCT e.type AS entity_labels, r.type AS relationship_type
              """, user_name=user_name)
            entities = []
            relationships = []
            for record in result:
                entity_labels = record.get("entity_labels")
                relationship_type = record.get("relationship_type")
                if entity_labels:
                    entities.append(entity_labels)
                if relationship_type:
                    relationships.append(relationship_type)
        return entities, relationships

    def execute_cypher_query(self, query):
        with self.driver.session() as session:
            result = session.run(query)
            formatted_results = []
            for record in result:
                path1 = record.get('path1')
                path2 = record.get('path2')
                sentence = ""
                def process_path(path, path_name, end):
                    path_description = []
                    if path:
                        nodes = path.nodes
                        relationships = path.relationships
                        for i in range(len(nodes)-end):
                            node = nodes[i]
                            name = node.get('name')
                            entity_type = node.get('type')
                            path_description.append(f"{name} ({entity_type})")
                            if i < len(relationships):
                                relationship = relationships[i]
                                relationship_type = relationship.get('type')
                                path_description.append(f"--[{relationship_type}]-->")
                    path_sentence = " ".join(path_description)
                    return path_sentence
                if path2: sentence = sentence+process_path(path1, "Path1", 1)
                else: sentence = sentence+process_path(path1, "Path1", 0)
                sentence = sentence+process_path(path2, "Path2", 0)
                formatted_results.append(sentence)
            return formatted_results
