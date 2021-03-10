import firebase_admin
from firebase_admin import credentials, firestore
import time

class Customer(object):
    def __init__(self, face_id, age, ethnicity, gender, visits, probabilities, inLine, last_updated):
        self.fac_id = face_id
        self.ethnicity = ethnicity
        self.gender = gender
        self.visits = visits
        self.probabilities = probabilities
        self.inLine = inLine
        self.last_updated = last_updated

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)


order_list = [{'value': 2, 'name': "Grilled Chicken Sandwich"}, {'value': 4, 'name': "Spicy Chicken Sandwich"},{'value': 1, 'name': "Nuggets"}, {'value': 1, 'name': "Waffle Potato Fries"}]

db = firestore.client()
doc_ref = db.collection(u'Expo_Customers').document(u'fb2a3461-db5b-4c44-8bde-e521c1298560')
'''doc_ref.set({
    u'face_id': u'Joel',
    u'age': 23,
    u'ethnicity': u'white',
    u'gender': u'male',
    u'visits': 6,
    u'probabilities': order_list,
    u'inLine': False,
    u'last_updated': int(round(time.time() * 1000))
})'''
doc_ref.update({u'inLine': False})

col_ref = db.collection(u'Expo_Customers').order_by(u'last_updated').get()
names = []

for doc in col_ref:
    #print(doc.to_dict()['face_id'])
    names = names + [doc.to_dict()['face_id']]

print(names)
