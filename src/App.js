import {BrowserRouter, Route, Routes} from "react-router-dom";

function App() {

    return (<div className="container">
        <BrowserRouter>
            <div className="row mt-2">
                <div className="col-2">
                    <h1>Test shit</h1>
                </div>
                <div className="col-10">
                    {/* <Routes>
                        <Route path="/"
                               element={<HomeComponent/>}/>
                    </Routes> */}
                    <h1>Hello World</h1>
                </div>
            </div>
        </BrowserRouter>
    </div>);
}

export default App;