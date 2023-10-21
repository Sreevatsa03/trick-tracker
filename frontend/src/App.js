import "./styles.css";
import LiveWebcam from "./components/LiveWebcam";
import TrickDetected from "./components/TrickDetected";

export default function App() {
  return (
    <div className="container text-center">
      <div className="row">
        <div className="col-sm-12">
          <header className="my-4">
            <h1>Welcome to Trick Tracker!</h1>
          </header>
        </div>
      </div>
      <div className="row">
        <div className="col-sm-8 mx-auto">
          <main className="my-4">
            <LiveWebcam />
          </main>
        </div>
      </div>
      <div className="row">
        <div className="col-sm-12">
          <footer className="my-4">
            <TrickDetected />
          </footer>
        </div>
      </div>
    </div>
  );
}
